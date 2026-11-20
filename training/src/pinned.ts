import { execFileSync } from 'node:child_process'
import { createHash } from 'node:crypto'
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import process from 'node:process'
import { cacheDir, dataDir, readJSON } from './paths'

interface GitSource {
  type: 'git'
  repo: string
  commit: string
  /** Sparse checkout patterns, so that only these files download */
  paths: string[]
  license: string
}

interface FileSource {
  type: 'file'
  url: string
  /** Empty until the first download with `--pin` */
  sha256: string
  license: string
}

type Source = GitSource | FileSource

const manifestFile = join(dataDir, 'sources.json')

function loadManifest(): Record<string, Source> {
  return readJSON<Record<string, Source>>(manifestFile)
}

function git(cwd: string, ...args: string[]): void {
  execFileSync('git', args, { cwd, stdio: 'ignore' })
}

function fetchGit(id: string, source: GitSource): string {
  const dir = join(cacheDir, 'pinned', `${id}-${source.commit.slice(0, 12)}`)
  if (existsSync(join(dir, '.done')))
    return dir
  mkdirSync(dir, { recursive: true })
  git(dir, 'init', '-q')
  git(dir, 'remote', 'add', 'origin', source.repo)
  git(dir, 'sparse-checkout', 'set', '--no-cone', ...source.paths)
  git(dir, 'fetch', '-q', '--depth', '1', '--filter=blob:none', 'origin', source.commit)
  git(dir, 'checkout', '-q', 'FETCH_HEAD')
  writeFileSync(join(dir, '.done'), '')
  return dir
}

async function fetchFile(id: string, source: FileSource, pin: boolean): Promise<string> {
  const file = join(cacheDir, 'pinned', `${id}.json`)
  if (!existsSync(file)) {
    const response = await fetch(source.url)
    if (!response.ok)
      throw new Error(`Download of ${id} failed with HTTP ${response.status}`)
    mkdirSync(join(cacheDir, 'pinned'), { recursive: true })
    writeFileSync(file, new Uint8Array(await response.arrayBuffer()))
  }
  const sha256 = createHash('sha256').update(readFileSync(file)).digest('hex')
  if (pin)
    source.sha256 = sha256
  else if (source.sha256 !== sha256)
    throw new Error(`Hash of ${id} is ${sha256}, but training/data/sources.json expects ${source.sha256 || 'no hash'}. Run the data script with --pin to accept the new file.`)
  return file
}

/**
 * Download each pinned source into the cache, and return its local path.
 * With `pin`, the hashes of the downloaded files replace the hashes in the manifest.
 */
export async function fetchPinned(pin = process.argv.includes('--pin')): Promise<Record<string, string>> {
  const manifest = loadManifest()
  const paths: Record<string, string> = {}
  for (const [id, source] of Object.entries(manifest)) {
    paths[id] = source.type === 'git'
      ? fetchGit(id, source)
      : await fetchFile(id, source, pin)
  }
  if (pin)
    writeFileSync(manifestFile, `${JSON.stringify(manifest, null, 2)}\n`)
  return paths
}
