
import Foundation

func interfaceIntTime(second: Int64) -> String {
    var input = second
    let hours: Int64 = input / 3600
    input = input % 3600
    let mins: Int64 = input / 60
    let secs: Int64 = input % 60
    
    guard second >= 0 else {
        fatalError("Second can not be negative: \(second)");
    }
    
    return String(format: "%02d:%02d:%02d", hours, mins, secs)
}

func timeToString() -> String {
    let date = Date()
    let calendar = Calendar.current
    let year = calendar.component(.year, from: date)
    let month = calendar.component(.month, from: date)
    let day = calendar.component(.day, from: date)
    let hour = calendar.component(.hour, from: date)
    let minute = calendar.component(.minute, from: date)
    let sec = calendar.component(.second, from: date)
    return String(format:"%04d-%02d-%02d %02d:%02d:%02d", year, month, day, hour, minute, sec)
}
