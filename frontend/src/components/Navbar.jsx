import { Menu, MessageSquare } from "lucide-react";
import { useLocation } from "react-router-dom";

export default function Navbar({ onMenuClick, onChatClick }) {
  const { pathname } = useLocation();
  const isChatbot = pathname.startsWith("/chatbot");

  return (
    <div className="w-full bg-white border-b h-16 flex items-center px-4 gap-3">
      <button className="md:hidden" onClick={onMenuClick}>
        <Menu size={22} />
      </button>

      {isChatbot && (
        <button className="md:hidden" onClick={onChatClick}>
          <MessageSquare size={22} />
        </button>
      )}

      <h2 className="font-semibold text-lg ml-2">PrediX AI</h2>
    </div>
  );
}
