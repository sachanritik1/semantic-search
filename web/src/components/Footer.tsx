export default function Footer() {
	const year = new Date().getFullYear();

	return (
		<footer className="mt-16 border-t border-[var(--line)] px-4 pb-10 pt-8 text-[var(--sea-ink-soft)]">
			<div className="page-wrap text-center text-sm sm:text-left">
				<p className="m-0">
					&copy; {year} Semantic Search. Hybrid RAG over your documents.
				</p>
			</div>
		</footer>
	);
}
