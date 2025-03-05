export const genstudio = {
  // Registry of all component instances
  instances: {},
  // Hooks to run before PDF generation (e.g., for WebGPU canvas snapshot)
  beforePDFHooks: new Map(),
  // Hooks to run after PDF generation (e.g., for cleanup)
  afterPDFHooks: new Map()
}

genstudio.whenReady = async function(id) {
  while (!genstudio.instances[id]) {
    await new Promise(resolve => setTimeout(resolve, 50));
  }
  await genstudio.instances[id].whenReady();
};

/**
 * Run all registered beforePDF hooks - used to prepare special content
 * like WebGPU canvases before PDF capture
 */
genstudio.beforePDF = async function(id) {
  await genstudio.whenReady(id);
  const hooks = Array.from(genstudio.beforePDFHooks.values())
    .filter(hook => hook && typeof hook === 'function');

  if (hooks.length > 0) {
    await Promise.all(hooks.map(hook => hook(id)));
  }
};

/**
 * Run all registered afterPDF hooks - used to clean up resources
 * after PDF capture is complete
 */
genstudio.afterPDF = async function(id) {
  const hooks = Array.from(genstudio.afterPDFHooks.values())
    .filter(hook => hook && typeof hook === 'function');

  if (hooks.length > 0) {
    await Promise.all(hooks.map(hook => hook(id)));
  }
};

window.genstudio = genstudio
