from oculus_reader import OculusReader

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Utility to manage teleoperation APK. Installs APK if no arguments are provided.')
    parser.add_argument("--reinstall", action="store_true", help='reinstalls APK from the default path')
    parser.add_argument("--uninstall", action="store_true", help='uninstalls APK')
    args = parser.parse_args()

    reader = OculusReader(run=False)

    if args.reinstall:
        reader.install(reinstall=True)
    elif args.uninstall:
        reader.uninstall()
    else:
        reader.install()
    print('Done.')

if __name__ == "__main__":
    main()
