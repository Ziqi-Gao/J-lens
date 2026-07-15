(function initializeReportRegistry(global) {
  "use strict";

  const directions = new Map([
    ["concept", []],
    ["jspace", []],
  ]);

  global.JLensReportRegistry = Object.freeze({
    register(direction, run) {
      if (!directions.has(direction)) {
        throw new Error(`Unknown report direction: ${direction}`);
      }
      if (!run || typeof run.id !== "string" || run.id.length === 0) {
        throw new Error("Every report run needs a stable string id.");
      }
      const runs = directions.get(direction);
      if (runs.some((candidate) => candidate.id === run.id)) {
        throw new Error(`Duplicate report run id: ${run.id}`);
      }
      runs.push(Object.freeze(run));
    },

    get(direction) {
      return [...(directions.get(direction) || [])];
    },
  });
})(window);
