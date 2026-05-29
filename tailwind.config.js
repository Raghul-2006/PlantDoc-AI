/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: "#154212",
        "primary-container": "#2d5a27",
        secondary: "#7c5730",
        background: "#fafaf5",
        "surface-1": "#f4f4ef",
        "surface-2": "#eeeee9",
        "accent-alert": "#8d3220",
        "on-surface": "#1a1c19",
        "on-primary": "#ffffff",
        "outline-variant": "#c2c9bb",
      },
      fontFamily: {
        headline: ["Manrope", "sans-serif"],
        body: ["Inter", "sans-serif"],
      },
      borderRadius: {
        'xl': '1rem',
        '2xl': '1.5rem',
        '3xl': '2rem',
      },
    },
  },
  plugins: [],
}
