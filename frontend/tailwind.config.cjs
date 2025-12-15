/** @type {import('tailwindcss').Config} */
module.exports = {
content: [
  "./index.html",
  "./src/**/*.{js,jsx,ts,tsx}",
],
  theme: {
    extend: {
      animation: {
        floatSlow: "float 7s ease-in-out infinite",
        floatFast: "float 4.5s ease-in-out infinite",
},
      keyframes: {
        float: {
          "0%, 100%": { transform: "translateY(0px)" },
          "50%": { transform: "translateY(-40px)" },
        },
      },
    },
  },
  plugins: [],
};
