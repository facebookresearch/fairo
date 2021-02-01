import React from "react";

export default function MemoryManager({
  memories,
  named_abstractions,
  reference_objects,
  triples,
}) {
  function getCount() {
    return reference_objects.length;
  }

  function getMemory(index) {
    return reference_objects[index];
  }

  return [getCount, getMemory];
}
