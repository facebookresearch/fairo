export default function MemoryManager(
  { memories, named_abstractions, reference_objects, triples },
  filter
) {
  // TODO create a lookup of memory from uuid
  const lookup = new Map(reference_objects.map((data) => [data[0], { data }]));

  if (triples) {
    triples.forEach((triple) => {
      // index = 1 for lookup Id.
      const subjUUID = triple[1];

      const entry = lookup.get(subjUUID);

      if (!entry) {
        // console.log("missing Entry: ", subjUUID);
        return;
      }

      if (!entry.triples) {
        entry.triples = [triple];
      } else {
        entry.triples.push(triple);
      }
    });
  }

  var filtered_refrence_objects = [];
  // build out a list of filtered objects
  if (null != filter && filter.length > 0) {
    debugger;
    reference_objects.forEach((ref_object) => {
      // Check the name
      if (JSON.stringify({ ...ref_object }).includes(filter)) {
        debugger;
        filtered_refrence_objects.push(ref_object);
      }
    });
  } else {
    filtered_refrence_objects = reference_objects;
  }

  // TODO create a search lookup of memories to find by id.

  function getCount() {
    return filtered_refrence_objects.length;
  }

  function getMemoryForIndex(index) {
    return filtered_refrence_objects[index];
  }

  function getMemoryForUUID(uuid) {
    return lookup.get(uuid);
  }

  return [getCount, getMemoryForIndex, getMemoryForUUID];
}
