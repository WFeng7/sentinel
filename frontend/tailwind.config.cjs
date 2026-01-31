/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./index.html",
    "./src/**/*.{js,jsx}"
  ],
  theme: {
    extend: {
      colors: {
        "bg-primary": "var(--bg-primary)",
        "text-primary": "var(--text-primary)",
        "button-primary": "var(--button-primary)",
        "metadata-1": "var(--metadata-1)",
        "metadata-2": "var(--metadata-2)",
        "metadata-3": "var(--metadata-3)",
        "metadata-4": "var(--metadata-4)"
      }
    }
  },
  plugins: []
}
