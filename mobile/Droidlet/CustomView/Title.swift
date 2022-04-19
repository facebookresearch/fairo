import SwiftUI

struct Title: ViewModifier {
    var enable = false
    func body(content: Content) -> some View {
        content
            .foregroundColor(enable ? Color.blue : Color.gray)
    }
}

extension View {
    func onNotification(
        _ notificationName: Notification.Name,
        perform action: @escaping () -> Void
    ) -> some View {
        onReceive(NotificationCenter.default.publisher(
            for: notificationName
        )) { _ in
            action()
        }
    }

    func onAppEnteredBackground(
        perform action: @escaping () -> Void
    ) -> some View {
        onNotification(
            UIApplication.didEnterBackgroundNotification,
            perform: action
        )
    }
}
