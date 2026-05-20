/** Parsed citation from markers like 【3†source=file.pdf&page=3】 */
export interface ParsedCitation {
	index: number;
	source: string;
	page?: string;
}

const CITATION_PATTERN = /【(\d+)[†‡|]([^】]+)】/g;

export function parseCitationBody(
	body: string,
): Pick<ParsedCitation, "source" | "page"> {
	const sourceMatch = body.match(/(?:^|&)source=([^&]+)/);
	const pageMatch = body.match(/(?:^|&)page=([^&]+)/);
	return {
		source: sourceMatch?.[1]?.trim() ?? body.trim(),
		page: pageMatch?.[1]?.trim(),
	};
}

export function parseCiteHref(href: string): ParsedCitation | null {
	if (!href.startsWith("cite:")) {
		return null;
	}
	const rest = href.slice(5);
	const queryIndex = rest.indexOf("?");
	const indexPart = queryIndex === -1 ? rest : rest.slice(0, queryIndex);
	const index = Number.parseInt(indexPart, 10);
	if (Number.isNaN(index)) {
		return null;
	}
	const params = new URLSearchParams(
		queryIndex === -1 ? "" : rest.slice(queryIndex + 1),
	);
	const source = params.get("source")?.trim();
	if (!source) {
		return null;
	}
	const page = params.get("page")?.trim();
	return { index, source, page: page || undefined };
}

export function citationToMarkdownLink(citation: ParsedCitation): string {
	const params = new URLSearchParams({ source: citation.source });
	if (citation.page) {
		params.set("page", citation.page);
	}
	return `[${citation.index}](cite:${citation.index}?${params.toString()})`;
}

/** Convert LLM answer text (citations + loose lists) into GFM-friendly markdown. */
export function prepareAnswerMarkdown(text: string): string {
	let result = text.replace(/\r\n/g, "\n").trim();
	if (!result) {
		return "";
	}

	// Some models separate list items with "//"
	result = result.replace(/\s*\/\/\s*/g, "\n\n");

	// GFM lists need a leading newline; models often emit "…sentence: * item"
	result = result.replace(/([.:!?])\s+(\*\s+)/g, "$1\n\n$2");
	result = result.replace(/([.:!?])\s+(-\s+)/g, "$1\n\n$2");

	result = result.replace(CITATION_PATTERN, (_match, index, body) => {
		const citation: ParsedCitation = {
			index: Number.parseInt(index ?? "0", 10),
			...parseCitationBody(body ?? ""),
		};
		return citationToMarkdownLink(citation);
	});

	return result;
}

export function extractCitations(text: string): ParsedCitation[] {
	const seen = new Set<string>();
	const citations: ParsedCitation[] = [];

	for (const match of text.matchAll(CITATION_PATTERN)) {
		const index = Number.parseInt(match[1] ?? "0", 10);
		const body = match[2] ?? "";
		const citation: ParsedCitation = { index, ...parseCitationBody(body) };
		const key = `${citation.index}:${citation.source}:${citation.page ?? ""}`;
		if (seen.has(key)) {
			continue;
		}
		seen.add(key);
		citations.push(citation);
	}

	return citations.sort((a, b) => a.index - b.index);
}
