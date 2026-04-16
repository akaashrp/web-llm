import * as tvmjs from "@mlc-ai/web-runtime";

export const SAMPLED_TOKEN_READBACK_RING_RUNTIME_HINT =
  'sample_readback_mode="ring_vector" requires an @mlc-ai/web-runtime build that exposes sampled-token readback ring APIs. ' +
  "Link web-llm-2 to tvm-profiling's profiling branch runtime (for example, " +
  '`"@mlc-ai/web-runtime": "file:/path/to/tvm-profiling/web"` in package.json).';

type DLDeviceWithSampledTokenReadbackRing = tvmjs.DLDevice & {
  createSampledTokenReadbackRing: (
    options: tvmjs.SampledTokenReadbackRingOptions,
  ) => tvmjs.SampledTokenReadbackRing;
};

export function supportsSampledTokenReadbackRingRuntime(
  device: tvmjs.DLDevice,
): device is DLDeviceWithSampledTokenReadbackRing {
  return (
    typeof (device as Partial<DLDeviceWithSampledTokenReadbackRing>)
      .createSampledTokenReadbackRing === "function"
  );
}

export function ensureSampledTokenReadbackRingRuntime(
  device: tvmjs.DLDevice,
): asserts device is DLDeviceWithSampledTokenReadbackRing {
  if (!supportsSampledTokenReadbackRingRuntime(device)) {
    throw new Error(SAMPLED_TOKEN_READBACK_RING_RUNTIME_HINT);
  }
}

export function createSampledTokenReadbackRing(
  device: tvmjs.DLDevice,
  options: tvmjs.SampledTokenReadbackRingOptions,
): tvmjs.SampledTokenReadbackRing {
  ensureSampledTokenReadbackRingRuntime(device);
  return device.createSampledTokenReadbackRing(options);
}
