import type { Components } from "react-markdown";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Badge } from "#/components/ui/badge.tsx";
import {
	extractCitations,
	parseCiteHref,
	prepareAnswerMarkdown,
	type ParsedCitation,
} from "#/lib/answer/citations.ts";

function CitationBadge({ citation }: { citation: ParsedCitation }) {
	const shortSource = citation.source.split("/").pop() ?? citation.source;
	const label = citation.page ? `p. ${citation.page}` : shortSource;

	return (
		<sup className="ml-0.5 inline-flex align-super">
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

const markdownComponents: Components = {
	a: ({ href, children, ...props }) => {
		const citation = href ? parseCiteHref(href) : null;
		if (citation) {
			return <CitationBadge citation={citation} />;
		}
		return (
			<a
				href={href}
				target="_blank"
				rel="noopener noreferrer"
				className="text-(--lagoon-deep) underline underline-offset-2"
				{...props}
			>
				{children}
			</a>
		);
	},
};

interface AnswerContentProps {
	content: string;
	embedded?: boolean;
}

export function AnswerContent({
	content,
	embedded = false,
}: AnswerContentProps) {
	const trimmed = content.trim();
	if (!trimmed) {
		return (
			<p className="m-0 text-sm text-(--sea-ink-soft)">No answer returned.</p>
		);
	}

	const markdown = prepareAnswerMarkdown(trimmed);
	const citations = extractCitations(trimmed);

	return (
		<div
			className={
				embedded
					? "flex flex-1 flex-col gap-4 overflow-y-auto"
					: "island-shell space-y-4"
			}
		>
			{embedded ? null : <p className="island-kicker m-0">Answer</p>}

			<div className="prose prose-sm max-w-none text-(--sea-ink) prose-headings:text-(--sea-ink) prose-headings:font-semibold prose-strong:text-(--sea-ink) prose-p:my-2 prose-li:my-1 prose-ul:my-2 prose-ol:my-2">
				<ReactMarkdown
					remarkPlugins={[remarkGfm]}
					components={markdownComponents}
				>
					{markdown}
				</ReactMarkdown>
			</div>

			{citations.length > 0 ? (
				<div className="border-t border-(--line) pt-3">
					<p className="m-0 mb-2 text-xs font-medium uppercase tracking-wide text-(--sea-ink-soft)">
						Sources
					</p>
					<ul className="m-0 flex list-none flex-wrap gap-2 p-0">
						{citations.map((citation) => (
							<li
								key={`${citation.index}-${citation.source}-${citation.page ?? ""}`}
							>
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
