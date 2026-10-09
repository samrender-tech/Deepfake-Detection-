/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Three verdict colours. Deliberately NOT red/green: a binary
        // red/green palette is itself an overclaim, and "inconclusive" has to
        // read as a real outcome rather than a failure state.
        authentic: { bg: "#0d2b1f", fg: "#6ee7a8", ring: "#14543a" },
        manipulated: { bg: "#2d1616", fg: "#fca5a5", ring: "#5b2020" },
        inconclusive: { bg: "#2a2410", fg: "#fcd34d", ring: "#584a16" },
      },
    },
  },
  plugins: [],
};
