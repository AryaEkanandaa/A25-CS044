import QuickActionCard from "./QuickActionCard";
import LogoPredix from "../../assets/logo.png";

export default function ChatWelcome({ userName, quickActions, setInput }) {
  return (
    <div className="w-full flex flex-col items-center px-4 py-10">
      {/* DESKTOP WIDTH  */}
      <div className="max-w-4xl w-full space-y-8">

        {/* LOGO */}
        <img
          src={LogoPredix}
          alt="PrediX Logo"
          className="mx-auto w-28 h-28 object-contain"
        />

        {/* TITLE */}
        <div className="text-center space-y-2">
          <h1
            className="
              text-4xl font-bold
              bg-gradient-to-r from-indigo-600 to-purple-600
              text-transparent bg-clip-text
            "
          >
            Hai, {userName}
          </h1>

          <p className="text-lg text-gray-600">
            Saya siap membantu Anda mengelola dan menganalisis sistem mesin
          </p>
        </div>

        {/* ================= QUICK ACTIONS ================= */}

        {/* MOBILE — horizontal scroll */}
        <div className="sm:hidden -mx-4 px-4">
          <div className="flex gap-4 overflow-x-auto pb-2">
            {quickActions.map((action, i) => (
              <div
                key={i}
                className="min-w-[280px] flex-shrink-0"
              >
                <QuickActionCard {...action} setInput={setInput} />
              </div>
            ))}
          </div>
        </div>

        {/* DESKTOP — GRID 2×3  */}
        <div className="hidden sm:grid grid-cols-2 gap-6">
          {quickActions.map((action, i) => (
            <QuickActionCard
              key={i}
              {...action}
              setInput={setInput}
            />
          ))}
        </div>

        {/* FOOTNOTE */}
        <p className="text-center text-sm text-gray-500 pt-8">
          Atau ketik pertanyaan Anda di bawah untuk memulai percakapan
        </p>

      </div>
    </div>
  );
}
