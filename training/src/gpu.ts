import { spawnSync } from 'node:child_process'
import { globSync } from 'node:fs'
import { homedir } from 'node:os'
import { dirname, join } from 'node:path'
import process from 'node:process'

/**
 * Find a folder with the SwiftShader software Vulkan driver.
 * Playwright ships it with Chromium, so a Playwright install is enough.
 */
function findSoftwareVulkan(): string | undefined {
  const roots = [
    process.env.PLAYWRIGHT_BROWSERS_PATH,
    '/opt/playwright',
    join(homedir(), '.cache/ms-playwright'),
  ]
  for (const root of roots) {
    const [manifest] = root
      ? globSync(join(root, '*/chrome-*/vk_swiftshader_icd.json'))
      : []
    if (manifest)
      return dirname(manifest)
  }
  return undefined
}

/**
 * On Linux without a GPU, restart this script with the software Vulkan driver on the library path.
 * The loader reads the library path only when a process starts, so a restart is the only way.
 * Call this first in every script that needs WebGPU.
 */
export function useSoftwareVulkanIfNeeded(): void {
  if (process.platform !== 'linux' || process.env.VK_ICD_FILENAMES)
    return
  const dir = findSoftwareVulkan()
  if (!dir)
    return
  const env = {
    ...process.env,
    VK_ICD_FILENAMES: join(dir, 'vk_swiftshader_icd.json'),
    LD_LIBRARY_PATH: [dir, process.env.LD_LIBRARY_PATH].filter(Boolean).join(':'),
  }
  const child = spawnSync(process.execPath, [
    ...process.execArgv,
    ...process.argv.slice(1),
  ], { env, stdio: 'inherit' })
  process.exit(child.status ?? 1)
}

/**
 * The Dawn instance. It must stay referenced for as long as a device is in use:
 * if the garbage collector frees it, the process crashes.
 */
let gpu: GPU | undefined

/** Open a WebGPU device in Node through Dawn: Metal on macOS, Vulkan on Linux */
export async function requestNodeDevice(): Promise<{ device: GPUDevice, adapter: string }> {
  const { create } = await import('webgpu')
  gpu ??= create([])
  const adapter = await gpu.requestAdapter()
  if (!adapter)
    throw new Error('WebGPU found no adapter. On Linux without a GPU, install Playwright Chromium for its software Vulkan driver.')
  const { vendor, architecture, description } = adapter.info
  return {
    device: await adapter.requestDevice(),
    adapter: [vendor, architecture, description].filter(Boolean).join(' '),
  }
}
