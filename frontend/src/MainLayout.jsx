import { useState } from "react";
import { useLocation, Outlet } from "react-router-dom";
import Sidebar from "./components/SideBar";
import Navbar from "./components/Navbar";

export default function MainLayout() {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const { pathname } = useLocation();

  const isChatbot = pathname.startsWith("/chatbot");
  const isMobile = window.innerWidth < 768;

  return (
    <div className="flex min-h-screen bg-gray-100">
      {sidebarOpen && (
        <div className="fixed inset-0 z-50 md:hidden">
          <div
            className="absolute inset-0 bg-black/40"
            onClick={() => setSidebarOpen(false)}
          />
          <div className="relative w-64 h-full bg-[#111827]">
            <Sidebar />
          </div>
        </div>
      )}

      <aside className="hidden md:block">
        <Sidebar />
      </aside>

      <div className="flex-1 flex flex-col min-w-0">
        {(!isChatbot || isMobile) && (
          <Navbar onMenuClick={() => setSidebarOpen(true)} />
        )}

        <div className="flex-1 overflow-hidden">
          <Outlet />
        </div>
      </div>
    </div>
  );
}
