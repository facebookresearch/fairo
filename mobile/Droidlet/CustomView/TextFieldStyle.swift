import Foundation
import SwiftUI

struct CustomTextFieldStyle: TextFieldStyle {
    @Binding var focused: Bool
    
    func _body(configuration: TextField<Self._Label>) -> some View {
        configuration
            .font(.callout)
            .foregroundColor(.white)
            .padding(10)
            .background(
                RoundedRectangle(cornerRadius: 4)
                    .strokeBorder(focused ? Color.red : Color.gray, lineWidth: 2)
            ).padding()
    }
}
