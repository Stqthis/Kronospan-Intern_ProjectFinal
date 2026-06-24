

if __name__ == "__main__":
    import multiprocessing

    # Required for frozen executables on Windows; harmless elsewhere.
    multiprocessing.freeze_support()

    from jarvisman.ui.gui import main

    main()