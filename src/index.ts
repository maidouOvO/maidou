// Basic TypeScript entry point
export type GreetingStyle = 'formal' | 'informal';

export const greet = (
  name: string,
  style: GreetingStyle = 'informal'
): string => {
  const greeting = style === 'formal' ? 'Good day' : 'Hello';
  return `${greeting}, ${name}!`;
};
