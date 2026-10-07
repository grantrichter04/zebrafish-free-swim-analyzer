"""
launch.pyw - what the "Free Swim Analyzer" desktop shortcut runs.

Loading the scientific libraries takes several seconds, and a shortcut has no
console, so without this nothing at all appears on screen in the meantime.
This shows a small "Starting" window first, and it deliberately imports
nothing from fish_analyzer until that window is visible.

Launched this way there is also nowhere for a startup error to be printed, so
one is shown in a dialog instead of the app silently failing to open.
"""
import tkinter as tk


def _show_splash() -> tk.Tk:
    splash = tk.Tk()
    splash.overrideredirect(True)
    splash.configure(bg="#1f4e79")
    tk.Label(splash, text="Free Swim Analyzer", font=("Arial", 20, "bold"),
             fg="white", bg="#1f4e79").pack(padx=50, pady=(28, 6))
    tk.Label(splash, text="Starting, this takes a few seconds...",
             font=("Arial", 11), fg="#d6e4f0", bg="#1f4e79").pack(pady=(0, 28))
    splash.update_idletasks()
    x = (splash.winfo_screenwidth() - splash.winfo_width()) // 2
    y = (splash.winfo_screenheight() - splash.winfo_height()) // 2
    splash.geometry(f"+{x}+{y}")
    splash.attributes("-topmost", True)
    splash.update()
    return splash


def main():
    splash = _show_splash()
    try:
        from fish_analyzer import EnhancedFishAnalyzer
    except Exception:
        import traceback
        from tkinter import messagebox
        splash.withdraw()
        messagebox.showerror(
            "Free Swim Analyzer could not start",
            "Run install.bat again. If this keeps happening, send this "
            "message to whoever looks after the laptop:\n\n"
            + traceback.format_exc()[-1500:])
        splash.destroy()
        return
    # The app makes its own main window; two at once confuse tkinter.
    splash.destroy()
    EnhancedFishAnalyzer().run()


if __name__ == "__main__":
    main()
