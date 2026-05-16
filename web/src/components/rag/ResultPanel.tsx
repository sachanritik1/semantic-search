export function ResultPanel({
	title,
	children,
}: {
	title: string;
	children: React.ReactNode;
}) {
	return (
		<div className="island-shell space-y-2">
			<p className="island-kicker m-0">{title}</p>
			<div className="font-mono text-sm leading-relaxed whitespace-pre-wrap text-[var(--sea-ink)]">
				{children}
			</div>
		</div>
	);
}
