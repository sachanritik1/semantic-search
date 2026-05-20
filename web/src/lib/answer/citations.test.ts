import { describe, expect, it } from "vitest";
import {
	extractCitations,
	parseCiteHref,
	prepareAnswerMarkdown,
} from "#/lib/answer/citations.ts";

const SAMPLE = `* Understanding the system's **requirements, constraints and bottlenecks**, which shape the overall direction of the design【3†source=System Design Interview by Alex Xu.pdf&page=3】
* Keeping services **stateless** to simplify scaling【4†source=System Design Interview by Alex Xu.pdf&page=148】`;

describe("prepareAnswerMarkdown", () => {
	it("converts citation markers to cite links and preserves bold", () => {
		const md = prepareAnswerMarkdown(
			"See **stateless** design【4†source=book.pdf&page=10】",
		);
		expect(md).toContain("**stateless**");
		expect(md).toContain("(cite:4?");
		expect(md).toContain("source=book.pdf");
		expect(md).toContain("page=10");
	});

	it("splits inline bullets onto new lines for GFM", () => {
		const md = prepareAnswerMarkdown(
			"The documents describe scaling: * Vertical scaling adds resources【2†source=book.pdf&page=9】 // * Horizontal scaling adds servers【2†source=book.pdf&page=9】",
		);
		expect(md).toContain("scaling:\n\n* Vertical");
		expect(md).toContain("\n\n* Horizontal");
	});

	it("parses bullet lists split by newlines or //", () => {
		const md = prepareAnswerMarkdown(SAMPLE);
		expect(md).toContain("(cite:3?");
		expect(md).toContain("(cite:4?");
	});
});

describe("parseCiteHref", () => {
	it("round-trips citation query params", () => {
		const citation = parseCiteHref("cite:4?source=book.pdf&page=10");
		expect(citation).toEqual({
			index: 4,
			source: "book.pdf",
			page: "10",
		});
	});
});

describe("extractCitations", () => {
	it("collects unique citations", () => {
		const citations = extractCitations(SAMPLE);
		expect(citations).toHaveLength(2);
		expect(citations[0]?.page).toBe("3");
		expect(citations[1]?.page).toBe("148");
	});
});
