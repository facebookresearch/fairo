import Foundation

/// A simple class to enhance debug logging.
class Logger {
    struct Constants {
        static let consoleLoggingEnabled = true
    }
    
    static func logDebug(_ debug: String) {
#if DEBUG
        print("Debug: \(debug)")
#endif
    }
}
