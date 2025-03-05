export const genstudio = {
  instances: {},
  beforePDFHooks: new Map(),
  afterPDFHooks: new Map()
}

genstudio.whenReady = async function(id) {
  while (!genstudio.instances[id]) {
    await new Promise(resolve => setTimeout(resolve, 50));
  }
  await genstudio.instances[id].whenReady();
};

genstudio.beforePDF = async function(id) {
  // Then run any registered before hooks
  await genstudio.whenReady(id);
  const hooks = Array.from(genstudio.beforePDFHooks.values())
    .filter(hook => hook && typeof hook === 'function');

  if (hooks.length > 0) {
    await Promise.all(hooks.map(hook => hook(id)));
  }
};

genstudio.afterPDF = async function(id) {
  // Run any registered after hooks
  const hooks = Array.from(genstudio.afterPDFHooks.values())
    .filter(hook => hook && typeof hook === 'function');

  if (hooks.length > 0) {
    await Promise.all(hooks.map(hook => hook(id)));
  }
};

window.genstudio = genstudio
