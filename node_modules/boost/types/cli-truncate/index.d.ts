declare module 'cli-truncate' {
  export interface CliTruncateOptions {
    position?: 'start' | 'middle' | 'end';
  }

  export default function cliTruncate(
    input: string,
    columns: number,
    options?: CliTruncateOptions,
  ): string;
}
