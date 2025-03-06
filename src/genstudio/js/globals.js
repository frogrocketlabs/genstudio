export const genstudio = {
  // Registry of all component instances
  instances: {}
}

genstudio.whenReady = async function(id) {
  while (!genstudio.instances[id]) {
    await new Promise(resolve => setTimeout(resolve, 50));
  }
  await genstudio.instances[id].whenReady();
};

window.genstudio = genstudio
