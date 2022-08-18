import Foundation
import SwiftUI

struct CustomButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .padding(.bottom)
            .foregroundColor(.white)
            .clipShape(Capsule())
    }
}

struct AccentFilledButton: ButtonStyle {
	@Environment(\.isEnabled) private var isEnabled

	func makeBody(configuration: Configuration) -> some View {
		configuration
			.label
			.foregroundColor(configuration.isPressed ? .gray : .white)
			.padding()
			.background(isEnabled ? Color.accentColor.opacity(0.8) : .gray.opacity(0.6))
			.cornerRadius(8)
	}
}
