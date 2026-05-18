import type { ReactNode } from "react";
import { cn } from "#/lib/utils.ts";

interface WorkspacePanelProps {
	kicker?: string;
	title: string;
	description?: string;
	children: ReactNode;
	className?: string;
}

export function WorkspacePanel({
	kicker,
	title,
	description,
	children,
	className,
}: WorkspacePanelProps) {
	return (
		<div
			className={cn(
				"island-shell flex min-h-[17.5rem] flex-col gap-4",
				className,
			)}
		>
			<div className="space-y-1">
				{kicker ? <p className="island-kicker m-0">{kicker}</p> : null}
				<h2 className="m-0 text-lg font-semibold text-(--sea-ink)">{title}</h2>
				{description ? (
					<p className="m-0 text-sm text-(--sea-ink-soft)">{description}</p>
				) : null}
			</div>
			<div className="flex flex-1 flex-col gap-4">{children}</div>
		</div>
	);
}
