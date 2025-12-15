export default function AnimatedBackground() {
    return (
        <div className="absolute inset-0 overflow-hidden z-0">
            <div className="absolute w-72 h-72 bg-indigo-500/20 rounded-full blur-3xl animate-floatSlow top-10 left-10" />
            <div className="absolute w-96 h-96 bg-purple-500/20 rounded-full blur-3xl animate-floatFast top-1/3 right-10" />
            <div className="absolute w-64 h-64 bg-cyan-400/20 rounded-full blur-3xl animate-floatSlow bottom-10 left-1/3" />
        </div>
    );
}
