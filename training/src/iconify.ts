import { join } from 'node:path'
import { packageDir, readJSON } from './paths'

interface IconSetInfo {
  name: string
  total?: number
  category?: string
  palette?: boolean
  hidden?: boolean
}

export interface IconSet {
  prefix: string
  info: IconSetInfo
  /** Icon names, hidden icons excluded */
  icons: string[]
  /** Alias name to the name of the icon it points to */
  aliases: Record<string, string>
  categories: Record<string, string[]>
}

interface RawIconSet {
  icons: Record<string, { hidden?: boolean }>
  aliases?: Record<string, { parent: string, hFlip?: boolean, vFlip?: boolean, rotate?: number }>
  categories?: Record<string, string[]>
}

const root = packageDir('@iconify/json')

/** Split an icon name or any text into lowercase words */
export function splitWords(text: string): string[] {
  return text.toLowerCase().match(/[a-z0-9]+/g) ?? []
}

export function loadCollections(): Record<string, IconSetInfo> {
  return readJSON<Record<string, IconSetInfo>>(join(root, 'collections.json'))
}

function loadIconSet(prefix: string, info: IconSetInfo): IconSet {
  const raw = readJSON<RawIconSet>(join(root, 'json', `${prefix}.json`))
  const aliases: Record<string, string> = {}
  for (const [alias, target] of Object.entries(raw.aliases ?? {})) {
    // A flipped or rotated alias is another picture, not another name
    if (!target.hFlip && !target.vFlip && !target.rotate)
      aliases[alias] = target.parent
  }
  return {
    prefix,
    info,
    icons: Object.keys(raw.icons).filter(name => !raw.icons[name]!.hidden),
    aliases,
    categories: raw.categories ?? {},
  }
}

/** Load every icon set that Iconify does not mark as hidden */
export function loadIconSets(): IconSet[] {
  return Object.entries(loadCollections())
    .filter(([, info]) => !info.hidden)
    .map(([prefix, info]) => loadIconSet(prefix, info))
}

export function iconifyVersion(): string {
  return readJSON<{ version: string }>(join(root, 'package.json')).version
}
