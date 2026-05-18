import { Badge } from "#/components/ui/badge.tsx";
import {
	collectCitations,
	parseAnswer,
	type InlineSegment,
	type ParsedCitation,
} from "#/lib/answer/parseAnswer.ts";

function CitationBadge({ citation }: { citation: ParsedCitation }) {
	const shortSource = citation.source.split("/").pop() ?? citation.source;
	const label = citation.page ? `p. ${citation.page}` : shortSource;

	return (
		<sup className="ml-0.5 align-super">
			<span
				className="inline-flex max-w-[12rem] cursor-default items-center gap-1 rounded-full border border-(--chip-line) bg-(--chip-bg) px-1.5 py-0.5 font-sans text-[0.65rem] font-medium leading-none text-(--lagoon-deep) no-underline"
				title={[citation.source, citation.page ? `page ${citation.page}` : null]
					.filter(Boolean)
					.join(" · ")}
			>
				<span className="tabular-nums text-(--sea-ink-soft)">
					{citation.index}
				</span>
				<span className="truncate">{label}</span>
			</span>
		</sup>
	);
}

function InlineContent({ segments }: { segments: InlineSegment[] }) {
	return (
		<>
			{segments.map((segment, index) => {
				if (segment.type === "bold") {
					return (
						<strong key={index} className="font-semibold text-(--sea-ink)">
							{segment.value}
						</strong>
					);
				}
				if (segment.type === "citation") {
					return <CitationBadge key={index} citation={segment.citation} />;
				}
				return <span key={index}>{segment.value}</span>;
			})}
		</>
	);
}

export function AnswerContent({ content }: { content: string }) {
	const blocks = parseAnswer(content);
	const citations = collectCitations(blocks);

	if (blocks.length === 0) {
		return (
			<div className="island-shell space-y-2">
				<p className="island-kicker m-0">Answer</p>
				<p className="m-0 text-sm text-(--sea-ink-soft)">No answer returned.</p>
			</div>
		);
	}

	return (
		<div className="island-shell space-y-4">
			<p className="island-kicker m-0">Answer</p>

			<div className="prose prose-sm max-w-none text-(--sea-ink) prose-headings:text-(--sea-ink) prose-strong:text-(--sea-ink) prose-p:my-2 prose-li:my-1">
				{blocks.map((block, blockIndex) => {
					if (block.type === "list") {
						return (
							<ul
								key={blockIndex}
								className="my-3 list-disc space-y-2 pl-5 marker:text-(--lagoon-deep)"
							>
								{block.items.map((item, itemIndex) => (
									<li key={itemIndex} className="leading-relaxed">
										<InlineContent segments={item.segments} />
									</li>
								))}
							</ul>
						);
					}

					return (
						<p key={blockIndex} className="m-0 leading-relaxed">
							<InlineContent segments={block.segments} />
						</p>
					);
				})}
			</div>

			{citations.length > 0 ? (
				<div className="border-t border-(--line) pt-3">
					<p className="m-0 mb-2 text-xs font-medium uppercase tracking-wide text-(--sea-ink-soft)">
						Sources
					</p>
					<ul className="m-0 flex flex-wrap gap-2 p-0 list-none">
						{citations.map((citation) => (
							<li key={`${citation.index}-${citation.source}-${citation.page ?? ""}`}>
								<Badge
									variant="secondary"
									className="max-w-xs truncate font-normal"
									title={citation.source}
								>
									<span className="mr-1 font-semibold tabular-nums">
										{citation.index}
									</span>
									{citation.source}
									{citation.page ? ` · p. ${citation.page}` : null}
								</Badge>
							</li>
						))}
					</ul>
				</div>
			) : null}
		</div>
	);
}
