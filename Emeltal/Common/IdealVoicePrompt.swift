import Foundation
import SwiftUI

struct IdealVoicePrompt: View {
    @Binding var shouldPromptForIdealVoice: Bool

    var body: some View {
        VStack(alignment: .leading) {
            HStack(alignment: .top) {
                Text("The ideal voice for this app is the premium variant of **Zoe**, which does not seem to be installed on your system. You can install it in system settings, and restart Emeltal for the best audio experience.")
                    .multilineTextAlignment(.leading)
                    .padding(6)

                Button {
                    withAnimation {
                        shouldPromptForIdealVoice = false
                    }
                } label: {
                    Image(systemName: "xmark")
                        .padding(6)
                }
            }

            Image(.voiceInstall)
                .resizable()
                .aspectRatio(contentMode: .fit)
                .padding(6)
        }
        .font(.body)
        .buttonStyle(.borderless)
        .padding(8)
        .foregroundColor(.black)
        .background(.accent)
        .clipShape(RoundedRectangle(cornerSize: CGSize(width: 16, height: 16), style: .continuous))
    }
}
