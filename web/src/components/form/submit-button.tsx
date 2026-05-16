import { useFormContext } from "#/hooks/form-context.ts";
import { Button } from "#/components/ui/button.tsx";
import { cn } from "#/lib/utils.ts";

export function SubmitButton({
	label,
	className,
	...props
}: React.ComponentProps<typeof Button> & { label: string }) {
	const form = useFormContext();

	return (
		<form.Subscribe selector={(state) => [state.isSubmitting, state.canSubmit]}>
			{([isSubmitting, canSubmit]) => (
				<Button
					type="submit"
					disabled={isSubmitting || !canSubmit || props.disabled}
					className={cn(className)}
					{...props}
				>
					{isSubmitting ? "Working…" : label}
				</Button>
			)}
		</form.Subscribe>
	);
}
