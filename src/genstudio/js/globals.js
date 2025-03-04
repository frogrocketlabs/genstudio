export const genstudio = {instances: {}}

genstudio.whenReady = async function(id) {
  while (!genstudio.instances[id]) {
    await new Promise(resolve => setTimeout(resolve, 50));
  }
  return genstudio.instances[id].whenReady();
};


window.genstudio = genstudio
