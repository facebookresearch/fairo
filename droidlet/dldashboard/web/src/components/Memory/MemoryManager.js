/**
 * Memory Manager simplifies the task of quering Agent's memory state.
 * @constructor
 *
 * @param {reference_objects} reference_objects The reference Objects in memory
 * @param {triples} triples All triples realted to the refrence Objects
 * @param {string} filter A fitler to apply when retreiving memory objecst.
 * @return {[function]} An array of functions that can be use to query Memory State
 *              @see getCount, @see getMemoryForIndex, @see getMemoryForUUID
 *
 */
export default function MemoryManager(
  { memories, named_abstractions, reference_objects, triples },
  filter
) {
  // reate a lookup of memory from uuid
  const lookup = new Map(reference_objects.map((data) => [data[0], { data }]));

  if (triples) {
    triples.forEach((triple) => {
      // index = 1 for lookup Id.
      const subjUUID = triple[1];

      const entry = lookup.get(subjUUID);

      if (!entry) {
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
    for (let value of lookup.values()) {
      // Check the name
      if (JSON.stringify(value).includes(filter)) {
        filtered_refrence_objects.push(value);
      }
    }
  } else {
    filtered_refrence_objects = Array.from(lookup.values());
  }

  /**
   * Returns the count of entires in memory
   */
  function getCount() {
    return filtered_refrence_objects.length;
  }
  /**
   * Returns the memory at index
   */
  function getMemoryForIndex(index) {
    return filtered_refrence_objects[index];
  }

  /**
   * Returns the memory for a given uuid.
   */
  function getMemoryForUUID(uuid) {
    return lookup.get(uuid);
  }

  return [getCount, getMemoryForIndex, getMemoryForUUID];
}
