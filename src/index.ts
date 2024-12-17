type GreetingStyle = "formal" | "informal";

export function createGreeting(name: string, style: GreetingStyle = "informal"): string {
  const greetings = {
    formal: `Good day, ${name}. How may I assist you?`,
    informal: `Hey ${name}! What's up?`
  };
  return greetings[style];
}
