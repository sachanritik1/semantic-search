import { describe, expect, it } from "vitest";
import {
	collectCitations,
	parseAnswer,
	parseInlineSegments,
} from "#/lib/answer/parseAnswer.ts";

const SAMPLE = `* Understanding the system's **requirements, constraints and bottlenecks**, which shape the overall direction of the design【3†source=System Design Interview by Alex Xu.pdf&page=3】
* Keeping services **stateless** to simplify scaling【4†source=System Design Interview by Alex Xu.pdf&page=148】`;

describe("parseInlineSegments", () => {
	it("parses bold and citation markers", () => {
		const segments = parseInlineSegments(
			"See **stateless** design【4†source=book.pdf&page=10】",
		);
		expect(segments).toEqual([
			{ type: "text", value: "See " },
			{ type: "bold", value: "stateless" },
			{ type: "text", value: " design" },
			{
				type: "citation",
				citation: { index: 4, source: "book.pdf", page: "10" },
			},
		]);
	});
});

describe("parseAnswer", () => {
	it("parses bullet lists split by newlines or //", () => {
		const blocks = parseAnswer(SAMPLE);
		expect(blocks).toHaveLength(1);
		expect(blocks[0]?.type).toBe("list");
		if (blocks[0]?.type === "list") {
			expect(blocks[0].items).toHaveLength(2);
		}
	});

	it("collects unique citations", () => {
		const blocks = parseAnswer(SAMPLE);
		const citations = collectCitations(blocks);
		expect(citations).toHaveLength(2);
		expect(citations[0]?.page).toBe("3");
		expect(citations[1]?.page).toBe("148");
	});
});
