export default function MemoryManager({
  memories,
  named_abstractions,
  reference_objects,
  triples,
}) {
  // TODO create a lookup of memory from uuid
  const lookup = new Map(reference_objects.map((data) => [data[0], data]));

  if (triples) {
    triples.forEach((triple) => {
      // index = 1 for lookup Id.
      const subjUUID = triple[1];

      const entry = lookup.get(subjUUID);

      if (!entry) {
        console.log("missing Entry: ", subjUUID);
        return;
      }

      if (!entry.triples) {
        entry.triples = [triple];
      } else {
        entry.triples.push(triple);
      }
    });
  }

  // TODO create a search lookup of memories to find by id.

  function getCount() {
    return reference_objects.length;
  }

  function getMemoryForIndex(index) {
    return reference_objects[index];
  }

  function getMemoryForUUID(uuid) {
    return lookup.get(uuid);
  }

  return [getCount, getMemoryForIndex, getMemoryForUUID];
}
