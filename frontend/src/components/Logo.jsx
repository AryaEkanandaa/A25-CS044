import logo from "../assets/logo.png";

export default function Logo() {
    return (
        <div className="flex justify-center mb-6">
            <img
                src={logo}
                alt="PrediX AI Logo"
                className="h-16 w-auto drop-shadow-lg"
            />
        </div>
    );
}
