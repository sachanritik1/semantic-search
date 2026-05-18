/** Parsed citation from markers like 【3†source=file.pdf&page=3】 */
export interface ParsedCitation {
	index: number;
	source: string;
	page?: string;
}

export type InlineSegment =
	| { type: "text"; value: string }
	| { type: "bold"; value: string }
	| { type: "citation"; citation: ParsedCitation };

export interface AnswerListItem {
	segments: InlineSegment[];
}

export type AnswerBlock =
	| { type: "paragraph"; segments: InlineSegment[] }
	| { type: "list"; items: AnswerListItem[] };

const CITATION_PATTERN = /【(\d+)[†‡|]([^】]+)】/g;
const BOLD_PATTERN = /\*\*(.+?)\*\*/g;
const LIST_LINE_PATTERN = /^\s*[*\-]\s+/;

function parseCitationBody(body: string): Pick<ParsedCitation, "source" | "page"> {
	const sourceMatch = body.match(/(?:^|&)source=([^&]+)/);
	const pageMatch = body.match(/(?:^|&)page=([^&]+)/);
	return {
		source: sourceMatch?.[1]?.trim() ?? body.trim(),
		page: pageMatch?.[1]?.trim(),
	};
}

export function parseInlineSegments(text: string): InlineSegment[] {
	if (!text) {
		return [];
	}

	const segments: InlineSegment[] = [];
	let lastIndex = 0;

	for (const match of text.matchAll(CITATION_PATTERN)) {
		const matchIndex = match.index ?? 0;
		if (matchIndex > lastIndex) {
			segments.push(...parseBoldSegments(text.slice(lastIndex, matchIndex)));
		}

		const index = Number.parseInt(match[1] ?? "0", 10);
		const body = match[2] ?? "";
		segments.push({
			type: "citation",
			citation: {
				index,
				...parseCitationBody(body),
			},
		});
		lastIndex = matchIndex + match[0].length;
	}

	if (lastIndex < text.length) {
		segments.push(...parseBoldSegments(text.slice(lastIndex)));
	}

	return segments.length > 0 ? segments : parseBoldSegments(text);
}

function parseBoldSegments(text: string): InlineSegment[] {
	if (!text) {
		return [];
	}

	const segments: InlineSegment[] = [];
	let lastIndex = 0;

	for (const match of text.matchAll(BOLD_PATTERN)) {
		const matchIndex = match.index ?? 0;
		if (matchIndex > lastIndex) {
			segments.push({ type: "text", value: text.slice(lastIndex, matchIndex) });
		}
		segments.push({ type: "bold", value: match[1] ?? "" });
		lastIndex = matchIndex + match[0].length;
	}

	if (lastIndex < text.length) {
		segments.push({ type: "text", value: text.slice(lastIndex) });
	}

	return segments.length > 0 ? segments : [{ type: "text", value: text }];
}

function normalizeAnswerText(text: string): string {
	return text
		.replace(/\r\n/g, "\n")
		.replace(/\s*\/\/\s*/g, "\n")
		.trim();
}

function isListLine(line: string): boolean {
	return LIST_LINE_PATTERN.test(line);
}

function stripListMarker(line: string): string {
	return line.replace(LIST_LINE_PATTERN, "").trim();
}

export function parseAnswer(text: string): AnswerBlock[] {
	const normalized = normalizeAnswerText(text);
	if (!normalized) {
		return [];
	}

	const lines = normalized.split("\n").map((line) => line.trim());
	const blocks: AnswerBlock[] = [];
	let listBuffer: AnswerListItem[] = [];

	const flushList = () => {
		if (listBuffer.length > 0) {
			blocks.push({ type: "list", items: listBuffer });
			listBuffer = [];
		}
	};

	for (const line of lines) {
		if (!line) {
			flushList();
			continue;
		}

		if (isListLine(line)) {
			listBuffer.push({
				segments: parseInlineSegments(stripListMarker(line)),
			});
			continue;
		}

		flushList();
		blocks.push({
			type: "paragraph",
			segments: parseInlineSegments(line),
		});
	}

	flushList();
	return blocks;
}

export function collectCitations(blocks: AnswerBlock[]): ParsedCitation[] {
	const seen = new Set<string>();
	const citations: ParsedCitation[] = [];

	const add = (citation: ParsedCitation) => {
		const key = `${citation.index}:${citation.source}:${citation.page ?? ""}`;
		if (seen.has(key)) {
			return;
		}
		seen.add(key);
		citations.push(citation);
	};

	for (const block of blocks) {
		const segments =
			block.type === "list"
				? block.items.flatMap((item) => item.segments)
				: block.segments;
		for (const segment of segments) {
			if (segment.type === "citation") {
				add(segment.citation);
			}
		}
	}

	return citations.sort((a, b) => a.index - b.index);
}
