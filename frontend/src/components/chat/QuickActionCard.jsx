export default function QuickActionCard({
  icon: Icon,
  title,
  description,
  template,
  prompt,
  setInput,
}) {
  const handleClick = () => {
    setInput(template ?? prompt ?? "");

    // Auto focus ke input chat (UX penting)
    requestAnimationFrame(() => {
      const el = document.querySelector("textarea, input[type='text']");
      el?.focus();
    });
  };

  return (
    <button
      type="button"
      onClick={handleClick}
      className="
        w-full text-left
        bg-white border border-gray-200
        rounded-xl
        p-4 sm:p-5
        transition
        active:scale-[0.98]
        hover:border-indigo-400 hover:shadow-md
        focus:outline-none focus:ring-2 focus:ring-indigo-400
      "
    >
      <div className="flex items-start gap-3">
        {/* ICON */}
        <div className="p-2.5 bg-indigo-50 rounded-lg">
          <Icon className="w-5 h-5 text-indigo-600" />
        </div>

        {/* TEXT */}
        <div className="flex-1">
          <h3 className="font-semibold text-gray-900 text-sm sm:text-base">
            {title}
          </h3>
          <p className="text-xs sm:text-sm text-gray-600 mt-0.5">
            {description}
          </p>
        </div>
      </div>
    </button>
  );
}
