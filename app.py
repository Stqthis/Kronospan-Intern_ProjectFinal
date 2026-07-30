

if __name__ == "__main__":
    import multiprocessing
    
    multiprocessing.freeze_support()

    from jarvisman.ui.gui import main

    main()