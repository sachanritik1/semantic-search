import { useFieldContext } from "#/hooks/form-context.ts";
import {
	Field,
	FieldDescription,
	FieldError,
	FieldLabel,
} from "#/components/ui/field.tsx";
import { Input } from "#/components/ui/input.tsx";
import {
	Select,
	SelectContent,
	SelectItem,
	SelectTrigger,
	SelectValue,
} from "#/components/ui/select.tsx";
import { Textarea } from "#/components/ui/textarea.tsx";

function useIsInvalid() {
	const field = useFieldContext();
	return field.state.meta.isTouched && !field.state.meta.isValid;
}

export function TextAreaField({
	label,
	placeholder,
	rows = 4,
	description,
}: {
	label: string;
	placeholder?: string;
	rows?: number;
	description?: string;
}) {
	const field = useFieldContext<string>();
	const isInvalid = useIsInvalid();

	return (
		<Field data-invalid={isInvalid}>
			<FieldLabel htmlFor={field.name}>{label}</FieldLabel>
			<Textarea
				id={field.name}
				name={field.name}
				value={field.state.value}
				onBlur={field.handleBlur}
				onChange={(event) => field.handleChange(event.target.value)}
				placeholder={placeholder}
				rows={rows}
				aria-invalid={isInvalid}
			/>
			{description ? <FieldDescription>{description}</FieldDescription> : null}
			{isInvalid ? <FieldError errors={field.state.meta.errors} /> : null}
		</Field>
	);
}

export function TextField({
	label,
	placeholder,
	description,
}: {
	label: string;
	placeholder?: string;
	description?: string;
}) {
	const field = useFieldContext<string>();
	const isInvalid = useIsInvalid();

	return (
		<Field data-invalid={isInvalid}>
			<FieldLabel htmlFor={field.name}>{label}</FieldLabel>
			<Input
				id={field.name}
				name={field.name}
				value={field.state.value}
				onBlur={field.handleBlur}
				onChange={(event) => field.handleChange(event.target.value)}
				placeholder={placeholder}
				aria-invalid={isInvalid}
			/>
			{description ? <FieldDescription>{description}</FieldDescription> : null}
			{isInvalid ? <FieldError errors={field.state.meta.errors} /> : null}
		</Field>
	);
}

export function FileField({
	label,
	accept,
	description,
}: {
	label: string;
	accept?: string;
	description?: string;
}) {
	const field = useFieldContext<File | null>();
	const isInvalid = useIsInvalid();

	return (
		<Field data-invalid={isInvalid}>
			<FieldLabel htmlFor={field.name}>{label}</FieldLabel>
			<input
				id={field.name}
				name={field.name}
				type="file"
				accept={accept}
				className="block w-full text-sm text-(--sea-ink) file:mr-4 file:cursor-pointer file:rounded-md file:border file:border-(--chip-line) file:bg-(--lagoon) file:px-4 file:py-2 file:text-sm file:font-semibold file:text-white file:transition-colors hover:file:bg-(--lagoon-deep) dark:file:border-(--chip-line) dark:file:bg-(--chip-bg) dark:file:text-(--sea-ink) dark:hover:file:bg-(--link-bg-hover)"
				aria-invalid={isInvalid}
				onBlur={field.handleBlur}
				onChange={(event) => {
					const selected = event.target.files?.[0] ?? null;
					if (
						selected &&
						accept?.includes(".pdf") &&
						!selected.name.toLowerCase().endsWith(".pdf")
					) {
						field.handleChange(null);
						return;
					}
					field.handleChange(selected);
				}}
			/>
			{description ? <FieldDescription>{description}</FieldDescription> : null}
			{isInvalid ? <FieldError errors={field.state.meta.errors} /> : null}
		</Field>
	);
}

export function SelectField({
	label,
	options,
	description,
}: {
	label: string;
	options: readonly string[];
	description?: string;
}) {
	const field = useFieldContext<string>();
	const isInvalid = useIsInvalid();

	return (
		<Field data-invalid={isInvalid}>
			<FieldLabel htmlFor={field.name}>{label}</FieldLabel>
			<Select
				value={field.state.value}
				onValueChange={(value) => field.handleChange(value)}
			>
				<SelectTrigger id={field.name} className="w-full">
					<SelectValue placeholder="Choose template" />
				</SelectTrigger>
				<SelectContent>
					{options.map((option) => (
						<SelectItem key={option} value={option}>
							{option}
						</SelectItem>
					))}
				</SelectContent>
			</Select>
			{description ? <FieldDescription>{description}</FieldDescription> : null}
			{isInvalid ? <FieldError errors={field.state.meta.errors} /> : null}
		</Field>
	);
}
