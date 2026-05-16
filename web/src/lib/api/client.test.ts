import { afterEach, describe, expect, it, vi } from "vitest";
import { apiPostFormData, apiPostJson } from "#/lib/api/client.ts";

describe("api client", () => {
	afterEach(() => {
		vi.unstubAllGlobals();
	});

	it("posts JSON to the given path", async () => {
		const fetchMock = vi.fn().mockResolvedValue({
			ok: true,
			status: 200,
			json: async () => ({ status: "ok" }),
		});
		vi.stubGlobal("fetch", fetchMock);

		await apiPostJson("/health", {});

		expect(fetchMock).toHaveBeenCalledWith(
			expect.stringContaining("/health"),
			expect.objectContaining({
				method: "POST",
				headers: { "Content-Type": "application/json" },
			}),
		);
	});

	it("uploads ingest files under the file field name", async () => {
		const fetchMock = vi.fn().mockResolvedValue({
			ok: true,
			status: 200,
			json: async () => ({
				document_id: "doc-1",
				chunks_total: 1,
				chunks_saved: 1,
			}),
		});
		vi.stubGlobal("fetch", fetchMock);

		const file = new File(["pdf"], "sample.pdf", { type: "application/pdf" });
		const formData = new FormData();
		formData.append("file", file);

		await apiPostFormData("/ingest", formData);

		const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
		const body = init.body as FormData;
		expect(body.get("file")).toBe(file);
	});
});
