import { readdirSync, readFileSync } from 'node:fs'
import { basename, join } from 'node:path'
import { parse as parseYaml } from 'yaml'
import { packageDir, readJSON } from './paths'

interface TaggedIcon {
  /** Id of the tag source, for example `lucide` */
  source: string
  /** Icon name in the source */
  name: string
  /** Search keywords that the icon set authors wrote for this icon */
  tags: string[]
}

type Loader = () => TaggedIcon[] | Promise<TaggedIcon[]>

function clean(source: string, name: string, tags: unknown): TaggedIcon {
  const list = Array.isArray(tags)
    ? tags
    : []
  return {
    source,
    name: name.toLowerCase(),
    tags: [
      ...new Set(list.map(tag => String(tag).toLowerCase().trim()).filter(tag => /[a-z]/.test(tag))),
    ],
  }
}

function lucide(): TaggedIcon[] {
  const tags = readJSON<Record<string, string[]>>(join(packageDir('lucide-static'), 'tags.json'))
  return Object.entries(tags).map(([name, list]) => clean('lucide', name, list))
}

function tabler(): TaggedIcon[] {
  const icons = readJSON<Record<string, { name: string, tags?: unknown[] }>>(join(packageDir('@tabler/icons'), 'icons.json'))
  return Object.values(icons).map(icon => clean('tabler', icon.name, icon.tags))
}

function mdi(): TaggedIcon[] {
  // The `tags` field of MDI holds categories. The `aliases` field holds the search keywords.
  const icons = readJSON<{ name: string, aliases: string[] }[]>(join(packageDir('@mdi/svg'), 'meta.json'))
  return icons.map(icon => clean('mdi', icon.name, icon.aliases))
}

async function phosphor(): Promise<TaggedIcon[]> {
  const { icons } = await import('@phosphor-icons/core')
  return icons.map(icon => clean('phosphor', icon.name, icon.tags))
}

function fontawesome(): TaggedIcon[] {
  const file = join(packageDir('@fortawesome/fontawesome-free'), 'metadata/icons.yml')
  const icons = parseYaml(readFileSync(file, 'utf8')) as Record<string, { search?: { terms?: unknown[] } }>
  return Object.entries(icons).map(([name, icon]) => clean('fontawesome', name, icon.search?.terms))
}

function octicons(): TaggedIcon[] {
  const icons = readJSON<Record<string, { name: string, keywords: string[] }>>(join(packageDir('@primer/octicons'), 'build/data.json'))
  return Object.values(icons).map(icon => clean('octicons', icon.name, icon.keywords))
}

function ionicons(): TaggedIcon[] {
  const { icons } = readJSON<{ icons: { name: string, tags: string[] }[] }>(join(packageDir('ionicons'), 'dist/ionicons.json'))
  return icons.map(icon => clean('ionicons', icon.name, icon.tags))
}

function carbon(): TaggedIcon[] {
  const { icons } = readJSON<{ icons: { name: string, aliases?: string[] }[] }>(join(packageDir('@carbon/icons'), 'metadata.json'))
  return icons.map(icon => clean('carbon', icon.name, icon.aliases))
}

function remix(file: string): TaggedIcon[] {
  const groups = readJSON<Record<string, Record<string, string> | string>>(file)
  return Object.values(groups)
    .filter(group => typeof group === 'object')
    .flatMap(group => Object.entries(group).map(([name, tags]) => clean('remix', name, tags.split(','))))
}

function feather(file: string): TaggedIcon[] {
  return Object.entries(readJSON<Record<string, string[]>>(file)).map(([name, tags]) => clean('feather', name, tags))
}

function materialSymbols(file: string): TaggedIcon[] {
  // Google prefixes the JSON with `)]}'` to stop direct script inclusion
  const text = readFileSync(file, 'utf8').replace(/^\)\]\}'/, '')
  const { icons } = JSON.parse(text) as { icons: { name: string, tags: string[] }[] }
  return icons.map(icon => clean('material-symbols', icon.name.replaceAll('_', '-'), icon.tags))
}

function bootstrap(dir: string): TaggedIcon[] {
  const folder = join(dir, 'docs/content/icons')
  return readdirSync(folder).filter(file => file.endsWith('.md')).map((file) => {
    const frontmatter = readFileSync(join(folder, file), 'utf8').split('---')[1] ?? ''
    const meta = parseYaml(frontmatter) as { tags?: unknown[] } | null
    return clean('bootstrap', basename(file, '.md'), meta?.tags)
  })
}

function fluent(dir: string): TaggedIcon[] {
  const folder = join(dir, 'assets')
  return readdirSync(folder).map((icon) => {
    const meta = readJSON<{ name: string, metaphor?: string[] }>(join(folder, icon, 'metadata.json'))
    return clean('fluent', meta.name.replaceAll(' ', '-'), meta.metaphor)
  })
}

/** Create one loader per tag source. `pinned` maps a pinned source id to its local path. */
export function tagLoaders(pinned: Record<string, string>): Record<string, Loader> {
  return {
    'lucide': lucide,
    'tabler': tabler,
    'mdi': mdi,
    'phosphor': phosphor,
    'fontawesome': fontawesome,
    'octicons': octicons,
    'ionicons': ionicons,
    'carbon': carbon,
    'remix': () => remix(pinned.remix!),
    'feather': () => feather(pinned.feather!),
    'material-symbols': () => materialSymbols(pinned['material-symbols']!),
    'bootstrap': () => bootstrap(pinned.bootstrap!),
    'fluent': () => fluent(pinned.fluent!),
  }
}
