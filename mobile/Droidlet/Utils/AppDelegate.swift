import UIKit
import SwiftUI
import GoogleSignIn
import Firebase
import FirebaseMessaging
import IQKeyboardManagerSwift

@main
class AppDelegate: UIResponder, UIApplicationDelegate {
    var settings: UserSettings = .shared
    var window: UIWindow?
    
    func application(_ application: UIApplication, didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?) -> Bool {
        
        print("Launch options: \(launchOptions as Optional)")
        let notificationOption = launchOptions?[.remoteNotification]
        print("Notification option: \(notificationOption as Optional)")
        
        let window = UIWindow(frame: UIScreen.main.bounds)
        window.rootViewController = UIHostingController(rootView: StartView())
        self.window = window
        window.makeKeyAndVisible()
        
        FirebaseApp.configure()
        
        Messaging.messaging().delegate = self
        UNUserNotificationCenter.current().delegate = self
        
        let authOptions: UNAuthorizationOptions = [.alert, .badge, .sound]
        UNUserNotificationCenter.current().requestAuthorization(
            options: authOptions,
            completionHandler: { _, _ in }
        )
        application.registerForRemoteNotifications()
        
        defineCustomAction()
        IQKeyboardManager.shared.enable = true
        return true
    }
    
    func applicationWillResignActive(_ application: UIApplication) {
        // Sent when the application is about to move from active to inactive state. This can occur for certain types of temporary interruptions (such as an incoming phone call or SMS message) or when the user quits the application and it begins the transition to the background state.
        // Use this method to pause ongoing tasks, disable timers, and invalidate graphics rendering callbacks. Games should use this method to pause the game.
    }
    
    func applicationDidEnterBackground(_ application: UIApplication) {
        // Use this method to release shared resources, save user data, invalidate timers, and store enough application state information to restore your application to its current state in case it is terminated later.
    }
    
    func applicationWillEnterForeground(_ application: UIApplication) {
        // Called as part of the transition from the background to the active state; here you can undo many of the changes made on entering the background.
    }
    
    func applicationDidBecomeActive(_ application: UIApplication) {
        // Restart any tasks that were paused (or not yet started) while the application was inactive. If the application was previously in the background, optionally refresh the user interface.
    }
    
    func application(_ app: UIApplication, open url: URL, options: [UIApplication.OpenURLOptionsKey : Any] = [:]) -> Bool {
		return GIDSignIn.sharedInstance.handle(url)
    }
    
}

extension AppDelegate: UNUserNotificationCenterDelegate, MessagingDelegate {
    func messaging(_ messaging: Messaging, didReceiveRegistrationToken fcmToken: String?) {
        print("Firebase registration token: \(String(describing: fcmToken))")
        
        let dataDict: [String: String] = ["token": fcmToken ?? ""]
        NotificationCenter.default.post(
            name: Notification.Name("FCMToken"),
            object: nil,
            userInfo: dataDict
        )
        // TODO: If necessary send token to application server.
        // Note: This callback is fired at each app startup and whenever a new token is generated.
        settings.deviceToken = fcmToken ?? ""
    }
    
    func application(application: UIApplication,
                     didRegisterForRemoteNotificationsWithDeviceToken deviceToken: Data) {
        Messaging.messaging().apnsToken = deviceToken
    }
    
    func userNotificationCenter(_ center: UNUserNotificationCenter, willPresent notification: UNNotification, withCompletionHandler completionHandler: @escaping (UNNotificationPresentationOptions) -> Void) {
        completionHandler([.banner, .sound])
    }
    
    func userNotificationCenter(_ center: UNUserNotificationCenter, didReceive response: UNNotificationResponse, withCompletionHandler completionHandler: @escaping () -> Void) {
        let content = response.notification.request.content
        let title = content.title
        let body = content.body
        
        let userInfo = content.userInfo
        let image_url = userInfo["image_url"] as? String
        let message_id = userInfo["message_id"] as? String
                
        // Perform the task associated with the action.
        switch response.actionIdentifier {
        case "ACCEPT_ACTION":
            settings.markImage(value: .yes, message_id: message_id ?? "", option: nil)
        case "DECLINE_ACTION":
            settings.notificationObject = NotificationObject(title: title, body: body, image_url: image_url ?? "", message_id: message_id ?? "")
            NotificationCenter.default.post(name: NSNotification.Name("Notification"), object: nil)
        default:
            break
        }
        
        // Always call the completion handler when done.
        completionHandler()
    }
    
    func defineCustomAction() {
        // Define the custom actions.
        let acceptAction = UNNotificationAction(identifier: "ACCEPT_ACTION",
                                                title: "Yes",
                                                options: [])
        let declineAction = UNNotificationAction(identifier: "DECLINE_ACTION",
                                                 title: "No",
                                                 options: [.foreground])
        // Define the notification type
        let meetingInviteCategory =
        UNNotificationCategory(identifier: "MEETING_INVITATION",
                               actions: [acceptAction, declineAction],
                               intentIdentifiers: [],
                               hiddenPreviewsBodyPlaceholder: "",
                               options: .customDismissAction)
        // Register the notification type.
        let notificationCenter = UNUserNotificationCenter.current()
        notificationCenter.setNotificationCategories([meetingInviteCategory])
    }
}
