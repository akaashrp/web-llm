import {
  ensureSampledTokenReadbackRingRuntime,
  supportsSampledTokenReadbackRingRuntime,
  SAMPLED_TOKEN_READBACK_RING_RUNTIME_HINT,
} from "../src/runtime_readback";

describe("runtime_readback compatibility helpers", () => {
  test("detects missing runtime support", () => {
    const baselineOnlyDevice = {} as any;
    expect(supportsSampledTokenReadbackRingRuntime(baselineOnlyDevice)).toBe(
      false,
    );
    expect(() =>
      ensureSampledTokenReadbackRingRuntime(baselineOnlyDevice),
    ).toThrow(SAMPLED_TOKEN_READBACK_RING_RUNTIME_HINT);
  });

  test("detects available runtime support", () => {
    const ringCapableDevice = {
      createSampledTokenReadbackRing: () => null,
    } as any;
    expect(supportsSampledTokenReadbackRingRuntime(ringCapableDevice)).toBe(
      true,
    );
    expect(() =>
      ensureSampledTokenReadbackRingRuntime(ringCapableDevice),
    ).not.toThrow();
  });
});
