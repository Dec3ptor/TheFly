// Parsers for the three binary formats the viewer reads.
//
//  * .flymesh  - the decimated neuropil shells built by tools/build_shells.py
//  * skeletons - neuroglancer "precomputed" skeletons, fetched live from Janelia's bucket
//  * .idx      - the packed neuron index built by tools/build_index.py

export const BUCKET = 'flyem-male-cns';
const SKELETONS = 'v1.0/segmentation/skeletons-malecns/skeletons-precomputed';

// Objects in the bucket are read through the GCS JSON API rather than the XML API:
// the JSON API sends CORS headers for every origin, so a static page on any host can
// read this data directly. The XML API (storage.googleapis.com/<bucket>/<path>) does
// not, and the browser would block it.
export function objectUrl(path) {
  return `https://storage.googleapis.com/storage/v1/b/${BUCKET}/o/` +
         `${encodeURIComponent(path)}?alt=media`;
}

/**
 * Decode a .flymesh shell into flat arrays ready for a BufferGeometry.
 * Coordinates are dequantised back to nanometres.
 */
export function parseFlymesh(buffer) {
  const view = new DataView(buffer);
  const magic = new TextDecoder().decode(new Uint8Array(buffer, 0, 8));
  if (magic !== 'FLYMESH1') throw new Error(`not a flymesh file (magic "${magic}")`);

  const vertexCount = view.getUint32(8, true);
  const triangleCount = view.getUint32(12, true);

  const origin = [view.getFloat32(16, true), view.getFloat32(20, true), view.getFloat32(24, true)];
  const scale = [view.getFloat32(28, true), view.getFloat32(32, true), view.getFloat32(36, true)];

  const quantised = new Uint16Array(buffer, 40, vertexCount * 3);
  const positions = new Float32Array(vertexCount * 3);
  for (let i = 0; i < vertexCount; i++) {
    positions[i * 3] = origin[0] + quantised[i * 3] * scale[0];
    positions[i * 3 + 1] = origin[1] + quantised[i * 3 + 1] * scale[1];
    positions[i * 3 + 2] = origin[2] + quantised[i * 3 + 2] * scale[2];
  }

  // The index block follows the vertices; slice() because its offset is not
  // guaranteed to satisfy Uint32Array's 4-byte alignment requirement.
  const indexStart = 40 + vertexCount * 6;
  const indices = new Uint32Array(buffer.slice(indexStart, indexStart + triangleCount * 12));

  return { positions, indices };
}

/**
 * Decode a neuroglancer precomputed skeleton: a vertex list plus the edges joining them.
 * Returned as a flat line-segment array, two endpoints per edge, in nanometres.
 */
export function parseSkeleton(buffer) {
  const view = new DataView(buffer);
  const vertexCount = view.getUint32(0, true);
  const edgeCount = view.getUint32(4, true);

  const expected = 8 + vertexCount * 12 + edgeCount * 8;
  if (buffer.byteLength < expected) {
    throw new Error(`truncated skeleton: ${buffer.byteLength} bytes, expected ${expected}`);
  }

  const vertices = new Float32Array(buffer.slice(8, 8 + vertexCount * 12));
  const edgeStart = 8 + vertexCount * 12;
  const edges = new Uint32Array(buffer.slice(edgeStart, edgeStart + edgeCount * 8));

  const segments = new Float32Array(edgeCount * 6);
  for (let i = 0; i < edgeCount; i++) {
    const a = edges[i * 2] * 3;
    const b = edges[i * 2 + 1] * 3;
    segments[i * 6] = vertices[a];
    segments[i * 6 + 1] = vertices[a + 1];
    segments[i * 6 + 2] = vertices[a + 2];
    segments[i * 6 + 3] = vertices[b];
    segments[i * 6 + 4] = vertices[b + 1];
    segments[i * 6 + 5] = vertices[b + 2];
  }
  return { segments, vertexCount, edgeCount };
}

/**
 * Fetch one neuron's skeleton. Not every segment has one; those return 404 and we
 * report that as null so the caller can skip the neuron rather than fail the batch.
 */
export async function fetchSkeleton(bodyId, signal) {
  const response = await fetch(objectUrl(`${SKELETONS}/${bodyId}`), { signal });
  if (response.status === 404) return null;
  if (!response.ok) throw new Error(`skeleton ${bodyId}: HTTP ${response.status}`);
  return parseSkeleton(await response.arrayBuffer());
}

/**
 * Decode the packed neuron index into a type-name table and a per-neuron record table.
 * Records arrive sorted by body id with the id delta-encoded, so they are rebuilt in
 * one pass. Parallel typed arrays keep 165k neurons cheap to hold.
 */
export function parseIndex(buffer) {
  const view = new DataView(buffer);
  const magic = new TextDecoder().decode(new Uint8Array(buffer, 0, 8));
  if (magic !== 'FLYIDX1\0') throw new Error(`not an index file (magic "${magic}")`);

  const typeCount = view.getUint32(8, true);
  const neuronCount = view.getUint32(12, true);
  const nameBytes = view.getUint32(16, true);

  const names = new TextDecoder()
    .decode(new Uint8Array(buffer, 20, nameBytes))
    .split('\n');
  if (names.length !== typeCount) {
    throw new Error(`index declares ${typeCount} types but carries ${names.length}`);
  }

  const bytes = new Uint8Array(buffer, 20 + nameBytes);
  let cursor = 0;
  const readVarint = () => {
    let value = 0;
    let shift = 0;
    for (;;) {
      const byte = bytes[cursor++];
      value += (byte & 0x7f) * 2 ** shift;
      if ((byte & 0x80) === 0) return value;
      shift += 7;
    }
  };

  const bodyIds = new Float64Array(neuronCount);   // ids exceed uint32 range in places
  const typeIndex = new Uint16Array(neuronCount);
  const synPre = new Uint32Array(neuronCount);
  const synPost = new Uint32Array(neuronCount);

  let previous = 0;
  for (let i = 0; i < neuronCount; i++) {
    previous += readVarint();
    bodyIds[i] = previous;
    typeIndex[i] = readVarint();
    synPre[i] = readVarint();
    synPost[i] = readVarint();
  }

  // Group neurons by type so a search result can list its members immediately.
  const membersByType = Array.from({ length: typeCount }, () => []);
  for (let i = 0; i < neuronCount; i++) membersByType[typeIndex[i]].push(i);

  return { names, membersByType, bodyIds, typeIndex, synPre, synPost, neuronCount };
}
