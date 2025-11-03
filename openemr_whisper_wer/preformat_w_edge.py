import pandas as pd
import re
from pathlib import Path

references = []

### GALLSTONES
ref_gallstones = """
Doctor: Good morning, please come in and take a seat. What brings you in today?
Patient: Morning, doctor. I’ve been having this severe pain in the upper right part of my stomach.
Doctor: oh Okay. Can you point exactly to where you feel the pain?
Patient: Around here, just under my ribs.
Doctor: oh okay Got it. And how long have you been experiencing this?
Patient: It’s been on and off for a few months, but in the past month the pain has become more frequent and much stronger.
Doctor: hmm When the pain comes, how would you describe it like is it sharp, dull, cramp-like?
Patient: feels like a stabbing pain, sometimes like a tight squeeze, and it radiates to my back at times.
Doctor: hmm That’s helpful. On a scale of 1 to 10, how bad is it at its worst?
Patient: Around 9. It’s almost unbearable.
Doctor:oh I can imagine that must be very uncomfortable. Have you noticed whether anything triggers it or makes it worse?
Patient: Yes, definitely after meals, especially if the food is oily or heavy.
Doctor: oh oily or heavy okay got it Does anything relieve the pain? okay does anything relieve the pain
Patient: Not really. I’ve tried some over-the-counter tablets for gastric upset, but they don’t seem to help.
Doctor: hmm Understood. Have you had any nausea, vomiting, or fever with these episodes?
Patient: No fever. Sometimes I feel a bit nauseous, but I haven’t vomited.
Doctor:hmm okay got it. Let me ask—have you noticed any changes in your bowel habits, or has your urine turned darker than usual?
Patient: Now that you mention it, my urine has looked darker lately. I thought it was just because I wasn’t drinking enough water.
Doctor: Hmm hmm. While we’re talking, I’m noticing a slight yellowish tinge in your eyes. Have you noticed that yourself?
Patient: (looks surprised) No, doctor, I didn’t pay attention. But looking now, yes, I can see it.
Doctor: That’s a mild sign of jaundice. It can happen if a gallstone is blocking the bile flow from the liver.
Patient: Oh, I see. I didn’t know that.
Doctor: Many people think stomach pain after eating is just “gastritis.” But gastritis pain is usually in the upper middle part of the abdomen, more like a burning sensation, and often improves with antacids. Your pain is in the upper right side, radiating to the back, worse after fatty meals as you mentionee, and you’ve got some jaundice—all of which point more toward gallstones rather than gastritis.
Patient: That makes sense now. Is it very serious?
Doctor: Gallstones can cause complications if untreated, like infection of the gallbladder or blockage of the bile ducts. But the good news is, if we diagnose it early, it’s very treatable.
Patient: Will I need surgery?
Doctor: If gallstones are confirmed and your symptoms are severe, the usual treatment is to remove the gallbladder surgically. It’s a common procedure and most people recover well. But first, we need to confirm the diagnosis.
Patient: Okay.
Doctor: I’ll order an abdominal ultrasound, which is the best test to detect gallstones. Along with that, I want you to get some blood tests—liver like function tests to check how your liver is working, and a cholesterol profile, since gallstones are often linked with cholesterol imbalance.
Patient: Alright, doctor.
Doctor: Before we finish, I want to check your records I see you have hypertension. Are you on medication, and how is it controlled?
Patient: Yes, I’m on medication, and my blood pressure is stable.
Doctor:oh Good good. Do you drink alcohol?
Patient: I used to drink heavily. These days I still drink often, but not as much as before.
Doctor: Thank you for being honest. Alcohol can strain the liver, so it’s best to cut it out cut it down further, especially with these symptoms.
Patient: I understand, I’ll work on that.
Doctor: Great. I’ll also do a quick abdominal examination now.
(Doctor palpates the abdomen gently.)
Doctor: oh okay I can feel some tenderness in the upper right quadrant, which matches your symptoms. That reinforces my suspicion of gallstones.
Patient: Okay.
Doctor: For now, avoid fatty, fried, and oily foods—they trigger gallbladder contractions and worsen the pain. Stay hydrated, and if the pain becomes unbearable, or if you develop fever, chills, or vomiting, you should come to the emergency room immediately.
Patient: I will, doctor. Thank you.
Doctor: You’re welcome. Let’s get those tests done, and once we have the results, we’ll discuss the next steps together.
"""

# Remove speaker labels only
ref_gallstones= re.sub(r'(?m)^(Doctor|Patient):\s*', '', ref_gallstones)

# Remove any text inside parentheses or brackets
ref_gallstones= re.sub(r'\([^)]*\)', '', ref_gallstones)
ref_gallstones= re.sub(r'\[[^\]]*\]', '', ref_gallstones)

# Convert multiple lines into a single line
ref_gallstones= re.sub(r'\s*\n\s*', ' ', ref_gallstones).strip()
references.append(ref_gallstones)

"-----------------------------------------------------------------------------------------------------"

### GERD
ref_gerd= """Doctor: Good morning. Please come in and have a seat. What brings you in today?
Patient: Morning, doctor. I’ve been vomiting a lot lately. I can barely keep any food down.
Doctor: I see. How long has this been happening?
Patient: It started a few weeks ago. At first, it was just nausea now and then, but now I vomit almost every day.
Doctor: And when you vomit, do you notice anything unusual? Blood, unusual color, or undigested food?
Patient: No no blood, just food and some stomach fluid.
Doctor: How about pain? Do you feel anything when vomiting or after meals?
Patient: Yes, I get this burning pain in my upper chest and stomach. It’s worse after meals.
Doctor: yeah Any trouble swallowing? Like food getting stuck?
Patient: Sometimes it feels a little tight going down.
Doctor: Any fever, chills, or recent weight changes besides what you mentioned?
Patient: No fever. I’ve lost some weight, but I thought it was from not eating much because of the vomiting.
Doctor: That makes sense. Can you tell me about your typical meals? Do certain foods make it worse?
Patient: okay I can barely eat, but fatty or heavy meals seem to trigger it more. so i have been trying to avoid fatty or heavy meals since few days
Doctor: Okay. Let’s talk about your medications. Can you tell me what you take daily for your diabetes, blood pressure, and or any other conditions?
Patient: i do have diabetes and i have blood pressure I take pills for both diabetes and my blood pressure. I think one of my medicines was changed recently, but I’m not sure which one.
Doctor: That’s helpful. Have you recently started any injections or new treatments?
Patient: I don’t really remember. Maybe something my old doctor mentioned
Doctor: Alright. I’ll check your records to be sure. (Doctor reviews medical chart.) okay yeah I see that you were recently started on a weekly injection for your diabetes, and the dose was increased about a month ago.
Patient: Oh, yes! I think I remember the nurse giving me an injection.
Doctor: That medication can help with diabetes and weight, but in some patients, it slows down stomach emptying, which can cause nausea and vomiting. In rare cases, repeated vomiting can injure the esophagus and cause severe reflux or ulcers.
Patient: I had no idea it could do that. No one told me it might cause this much trouble.
Doctor: It’s an uncommon complication, but it can happen, specially if vomiting is persistent. Since you’ve had this for a few weeks, I’d like to do some tests to check your esophagus.
Patient: What kind of tests?
Doctor: The main one is an endoscopy. It’s a procedure where we look at your esophagus and stomach with a small camera to see if there are ulcers or other damage. I’ll also start medications to reduce stomach acid and protect your esophagus.
Patient:oh yeah that sounds good. but Will it heal?
Doctor: Most people recover well with treatment. We’ll also pause the injection for now and monitor your blood sugar with other medications that don’t cause these side effects.
Patient: Okay. but And for my diabetes, is it safe to stop the injection?
Doctor: Yes, we’ll adjust your oral medications and monitor you closely. Once your stomach and esophagus have healed, we can decide the safest long-term plan for your diabetes.
Patient: I’m worried. Could this get worse if it’s not treated quickly?
Doctor: Yes, persistent vomiting can lead to dehydration, electrolyte imbalances, or even bleeding from the esophagus. That’s why we want to start treatment promptly.
Patient: oh yeah I understand. Is there anything I can do at home to feel better while waiting?
Doctor: Eat small, frequent meals, avoid lying down right after eating, and stick to soft or bland foods. Drink plenty of water, but in small sips. Avoid very fatty, spicy, or acidic foods until we see the results.
Patient: Okay, I’ll try that.
Doctor: Also, we’ll schedule a follow-up after the endoscopy. If you notice blood in your vomit, severe pain, or worsening vomiting, go to the emergency room immediately.
Patient: I will, doctor. Thank you for explaining everything. thank you very much
Doctor: You’re welcome. We’ll get the tests done soon and work together to manage your diabetes safely while taking care of your stomach."""
# Remove speaker labels only
ref_gerd= re.sub(r'(?m)^(Doctor|Patient):\s*', '', ref_gerd)

# Remove any text inside parentheses or brackets
ref_gerd= re.sub(r'\([^)]*\)', '', ref_gerd)
ref_gerd= re.sub(r'\[[^\]]*\]', '', ref_gerd)

# Convert multiple lines into a single line
ref_gerd= re.sub(r'\s*\n\s*', ' ', ref_gerd).strip()

references.append(ref_gerd)

"====================================================================================================================="

### Arthritis
ref_arthritis = """
Doctor: Good morning, Mr. Taylor. Come in and have a seat. How are you doing today?
Patient: Morning, doctor. Well, my knee’s been acting up again.
Doctor: Sorry to hear that. Which knee is giving you trouble this time?
Patient: It’s the right knee. It’s been bothering me more and more lately.
Doctor: Okay. When did you first start noticing it getting worse?
Patient: Hmm… probably about six months ago, but in the last couple of weeks it’s harder to walk any distance.
Doctor: I see. Can you describe the pain for me — is it sharp, dull, throbbing, or something like something else?
Patient: Mostly a dull ache, but sometimes sharp if I twist the wrong way.
Doctor: And if you had to rate it on a scale of 1 to 10?
Patient: Around a 6 after walking, maybe 3 if I’m just resting.
Doctor: oh Does the knee swell up?
Patient: Yes, especially after climbing stairs or walking a lot.
Doctor: aha Any redness or warmth?
Patient: No, it just looks a bit puffy.
Doctor: Has it ever locked, or felt like it was going to give way?
Patient: Once or twice, it felt like it might buckle, but it didn’t completely.
Doctor: Any clicking, popping, or grinding sensations?
Patient: Oh yes, lots of creaking when I bend it.
Doctor: And what have you tried for the pain so far?
Patient: Just over-the-counter medication like ibuprofen. It helps a bit but wears off quickly.
Doctor: Any stomach upset with that?
Patient: Not really, no.
Doctor: Good. Do you have any history of ulcers, bleeding problems, or kidney disease?
Patient: No, none of that.
Doctor: How about your general health? Any other medical conditions?
Patient: Well, I do have high blood pressure, but that’s controlled with tablets.
Doctor: Which tablets do you take?
Patient: Lisinopril, 10 milligrams once a day.
Doctor: And any allergies?
Patient: I think i am allergic to Penicillin.
Doctor: Okay, I’ll note that. Does arthritis run in your family?
Patient: Yes, my mum had bad knees in her 60s. She ended up needing surgery.
Doctor: Understood. And how has this been affecting your day-to-day life?
Patient: I can’t walk as far as I used to, and I’ve stopped playing golf. Even shopping is harder now — so I get sore halfway through.
Doctor: And what about sleep — does the pain disturb your rest?
Patient: Sometimes, yes. If I’ve been on my feet a lot, I’ll wake up in the night with an ache.
Doctor: Let’s take a look at that knee.
(Examines knee)
Doctor: I see some swelling around the joint. When I bend it, there’s a bit of creaking, what we call crepitus. Your movement is reduced compared to the other side, and there’s some tenderness here along the inside. But your ligaments feel stable, which is good.
Patient: So is that bad?
Doctor: It fits with osteoarthritis — basically wear and tear of the joint cartilage.
Patient: So nothing torn?
Doctor: Unlikely, based on the exam. But I’ll order an x-ray to confirm what’s going on inside the joint.
Doctor: okay, So, osteoarthritis is very common as we get older. There’s no cure as such, but there are many ways to manage symptoms and keep you mobile.
Patient: Should I stop walking then?
Doctor: No, actually staying active is one of the best things you can do. Keep walking, but pace yourself. Swimming and cycling are excellent because they’re easier on the joints.
Patient: I used to swim — I could try that again.
Doctor: That would be great. I’ll also refer you to physiotherapy. They’ll show you strengthening exercises for the thigh muscles, which support the knee. That can make a real difference.
Patient: Okay. Will I need stronger painkillers though?
Doctor: We can try naproxen, which is stronger than ibuprofen. To protect your stomach, I’ll give you a second tablet to take alongside it.
Patient: Alright.
Doctor: Do you drink alcohol?
Patient: Just a couple of beers on the weekend.
Doctor: That’s fine with naproxen, but always take it with food.
Patient: Got it.
Doctor: For swelling, you can also use ice packs after activity. And if things get worse, a walking stick or knee brace can help take pressure off.
Patient: I’d rather avoid a cane if I can.
Doctor: Understandable — we’ll keep that as a backup.
Patient: And if the tablets don’t work?
Doctor: Then we’d consider an injection into the joint, what which can help for several months. And if your mobility declines a lot, we can talk about referral to orthopaedics for surgery.
Patient: Like a knee replacement?
Doctor: Yes, though that’s usually only after other options are tried. You’re not at that stage yet.
Patient: That’s a relief.
Doctor: So, the plan is: x-ray, physio referral, naproxen plus stomach protector, and we’ll review in 3 months. In the meantime, keep active, try swimming again, and use ice if it swells.
Patient: Okay, thank you, doctor.
Doctor: You’re welcome. Take care, Mr. Taylor."""

# Remove speaker labels only
ref_arthritis= re.sub(r'(?m)^(Doctor|Patient):\s*', '', ref_arthritis)

# Remove any text inside parentheses or brackets
ref_arthritis= re.sub(r'\([^)]*\)', '', ref_arthritis)
ref_arthritis= re.sub(r'\[[^\]]*\]', '', ref_arthritis)

# Convert multiple lines into a single line
ref_arthritis= re.sub(r'\s*\n\s*', ' ', ref_arthritis).strip()

references.append(ref_arthritis)

"-----------------------------------------------------------------------------------------------------------------------------------"

### DIABETIC KETOACIDOSIS
ref_dka = """Provider: Hello there, I’m Dr. Lee. I hear you haven’t been feeling well. Can you tell me what’s been happening?
Patient: Hi, doctor. Yeah… my stomach hurts and I’ve been throwing up since yesterday. I just feel awful.
Provider: I’m sorry you’re feeling so poorly. When did the symptoms start?
Patient: Yesterday morning. First I felt nauseous, then I started vomiting. Since then, I’ve been so thirsty, drinking water constantly.
Provider: And how are you feeling right now?
Patient: Weak… tired. My parents say I’ve been a bit out of it today.
Provider: Okay. I see you’ve got type 1 diabetes, diagnosed about two years ago. Have you been taking your insulin as usual?
Patient: Normally yes, but I ran out two days ago and couldn’t get a refill.
Provider: Thanks for being honest — that’s really important. Skipping insulin can be dangerous. Have you checked your blood sugar at home?
Patient: No, I didn’t have any strips left.
Provider: Alright. Have you noticed fever, cough, or any other signs of infection?
Patient: No, its just the stomach pain and vomiting.
Provider: Let me examine you.
(examining)
Your heart rate is fast, your blood pressure is a bit low, and you’re breathing deeply and quickly. I also notice a fruity smell on your breath, I think this is typical when the body produces ketones.
Patient: Is that dangerous?
Provider: Yes, it can be. Based on your symptoms and what I see, I’m concerned you’re in diabetic ketoacidosis, or DKA. That means your body doesn’t have enough insulin, so it’s burning fat for fuel, producing acids called ketones. That explains your stomach pain, nausea, and the deep breathing.
Patient: So it’s because I missed insulin?
Provider: Exactly. Missing insulin is one of the most common causes. Sometimes infections or certain medications can also trigger it, but in your case, it looks like insulin omission costed.
Patient: That sounds serious.
Provider: It is, but the good news is we can treat it effectively if we act quickly. We’ve already I think we have already run some blood tests. Your glucose is very high at about 450, your blood is acidic with a pH of 7.24, and you have a lot of ketones in your system. I think That confirms the diagnosis.
Patient: What happens now?
Provider: Step one is IV fluids to correct dehydration — your kidneys are struggling a bit from fluid loss. Then we’ll give you insulin through a drip to steadily bring your sugar down and stop ketone production.
Patient: Okay.
Provider: One thing to understand is potassium. Even though your blood test shows its high, your body’s total potassium is actually low because you’ve lost a lot in urine. Once we give insulin, potassium shifts back into your cells, and your blood level can drop dangerously. So we’ll monitor it closely and give you potassium if it falls.
Patient: Wow… that sounds complicated.
Provider: It can be, but we’ll watch you carefully. We’ll check your sugar every hour, and your electrolytes and blood gases regularly.
Patient: do I have to stay in the hospital?
Provider: Yes. You’ll be admitted, usually for 24 to 48 hours, should be good. Once you’re stable, we’ll switch you back to injections instead of the drip.
Patient: I just have one question if I take my insulin then it won't happen again?
Provider: That’s right. The best prevention is taking your insulin every day. We’ll connect you with our diabetes educator to help you plan ahead and avoid gaps in medication.
Patient: What if I get sick like with the flu can this happen again?
Provider: Excellent question. Yes, illness can trigger DKA even if you’re taking insulin. That’s why we give “sick day rules” for people with diabetes never stop insulin, check sugars more often, stay hydrated, and seek help early if vomiting or unable to eat.
Patient: Will this affect me long-term?
Provider: If treated promptly, most people recover fully without long-term damage. But if untreated, DKA can be life-threatening especially because it can affect the brain and heart. That’s why it’s so important you came in when you did.
Patient: That’s a relief to hear.
Provider: So, here’s the plan we’ll start IV fluids and insulin right away, monitor you closely. Once you’re better, we’ll restart your regular insulin injections and review your diabetes care.
Patient: Okay, doctor. Thank you for explaining everything.
Provider: You’re welcome. We’ll take good care of you, and we’ll also make sure you leave with a clear plan for your insulin and diabetes management. Bye."""

# Remove speaker labels only
ref_dka= re.sub(r'(?m)^(Provider|Patient):\s*', '', ref_dka)

# Remove any text inside parentheses or brackets
ref_dka= re.sub(r'\([^)]*\)', '', ref_dka)
ref_dka= re.sub(r'\[[^\]]*\]', '', ref_dka)

# Convert multiple lines into a single line
ref_dka= re.sub(r'\s*\n\s*', ' ', ref_dka).strip()
references.append(ref_dka)

"---------------------------------------------------------------------------------------------------------------------------------"
ref_copd = """Doctor: Good morning, Ms. Sharma. How are you feeling today?
Patient: Morning, doctor. I’ve been having a bad cough and shortness of breath these past few days.
Doctor: I’m sorry to hear that. Can you tell me how long this has been happening?
Patient: It started about three days ago with a runny nose, and now I’m coughing more and feeling tired when I walk around.
Doctor: I see. Is the cough producing any phlegm or mucus?
Patient: Yes, it’s white and kind of thick. I usually have it in the mornings, but lately it’s worse.
Doctor: About how much phlegm are you bringing up in a day would you say it’s just a little, or quite a bit?
Patient: I’d say quite a bit, especially after I wake up. I have to cough for several minutes before I can breathe normally.
Doctor: Does the color ever change maybe yellow or green or has it stayed white?
Patient: Mostly white, maybe a little yellow when I get a cold, like this time.
Doctor: And have you noticed any blood when you cough?
Patient: No, never.
Doctor: Any fever, chills, or chest pain?
Patient: No, none of those. Just the cough and trouble breathing.
Doctor: Have you had similar problems before?
Patient: Yeah, I’ve been getting this cough on and off for years, and sometimes my breathing makes a little whistling noise.
Doctor: okay. Do you smoke, Ms. Sharma?
Patient: Yes, I’ve been smoking since my twenties. Around a pack and a half a day, sometimes two.
Doctor: So that would be, let’s see, close to forty years of smoking? That’s quite a long history of smoking
Patient: Yes, about that long. I’ve tried cutting down, but I haven’t quit completely.
Doctor: I understand it’s not easy. Do you notice that your cough gets worse after smoking or when you wake up?
Patient: Oh yes, mornings are the worst. I can barely get through my first cup of coffee without coughing.
Doctor: That’s pretty common in people with chronic bronchitis, which is one form of chronic obstructive pulmonary disease. The smoke irritates your airways, and over time the lungs produce extra mucus and become less efficient at moving air in and out.
Patient: That makes sense. I do feel like I’m always short of breath, especially going up stairs.
Patient: oh I see**.** It’s possible that your symptoms are early signs of COPD, but we’ll need to confirm that. There are a few other lung conditions that can cause similar symptoms, so I’d like to run some tests to be sure.
Patient: What kind of tests?
Doctor: First, I’d like to perform a spirometry test. You’ll take a deep breath and blow forcefully into a tube connected to a small machine. It measures how much air your lungs can hold and how quickly you can breathe out. This helps us see if your airways are narrowed.
Patient: Okay. Does it hurt?
Doctor: Not at all it’s completely painless and takes just a few minutes.
We’ll also do a pulse oximetry test, which is a small clip on your finger to check your oxygen levels. Depending on what we see, I might recommend a chest X ray to rule out infections or other causes, like asthma or heart-related problems.
Patient: Alright.
Doctor: Based on your history and the chronic cough, there’s a possibility that you’re developing COPD, but we can only diagnose that once the spirometry confirms airflow limitation.
Patient: I see. So what happens in the meantime?
Doctor: While we wait for the test results, I’d like you to star on a short-acting bronchodilator inhaler it helps open your airways and should make breathing easier, especially when you’re active or coughing a lot.
Patient: Will I need to use that every day?
Doctor: For now, only when you feel short of breath or having coughing fits. If the tests later confirm COPD or chronic bronchitis, we’ll plan a long-term treatment that might include daily inhalers or other medications.
Patient: Okay, that sounds reasonable.
Doctor: The most important thing right now is to cut down on smoking, and ideally quit altogether. It’s the single most effective step to prevent further lung damage.
Patient: I’ve tried quitting before, but it’s really tough.
Doctor: I understand it’s not easy. We can help you with nicotine patches, medications, and counseling programs that make it easier. Even reducing your cigarette count will start helping your lungs.
Patient: I’ll try again.
Doctor: That’s great. Also, try to stay hydrated , it helps loosen the mucus and avoid outdoor pollution or strong odors that can irritate your lungs.
Patient: Got it. Should I avoid exercise for now?
Doctor: Light activity like walking is fine, as long as you don’t push yourself too hard. Once your test results are back, we can discuss a pulmonary rehabilitation plan a supervised exercise and breathing program that strengthens your lungs safely.
Patient: Okay, that sounds good.
Doctor: Excellent. I’ll also check your oxygen levels today, and if needed, we might consider oxygen therapy later, but that depends entirely on your test results.
Patient: Alright.
Doctor: So, we’ll confirm the diagnosis only after the spirometry and imaging tests. In the meantime, use the inhaler as needed, drink plenty of fluids, and avoid irritants.
Patient: Thank you, doctor. You explained everything so clearly.
Doctor: You’re welcome, Ms. Sharma. I’ll have the nurse perform the spirometry now, and we’ll schedule your follow-up visit in one week to go over the results.
Patient: Sounds good. Thanks again.
Doctor: Take care, and I’ll see you soon."""

# Remove speaker labels only
ref_copd= re.sub(r'(?m)^(Doctor|Patient):\s*', '', ref_copd)

# Remove any text inside parentheses or brackets
ref_copd= re.sub(r'\([^)]*\)', '', ref_copd)
ref_copd= re.sub(r'\[[^\]]*\]', '', ref_copd)

# Convert multiple lines into a single line
ref_copd= re.sub(r'\s*\n\s*', ' ', ref_copd).strip()

references.append(ref_copd)

"---------------------------------------------------------------------------------------------------"
ref_anemia = """Doctor: Good morning, Ms. Sharma. I understand you’ve been feeling very tired and dizzy for past couple of weeks, and you also had a fall at home yesterday?
Patient: Yes, doctor. I didn’t hit my head or anything, just felt really weak and lightheaded.
Doctor: I’m glad you’re okay. Can you describe the dizziness? Does it happen when you stand, walk, or is it all the time?
Patient: huh Mostly when I stand or walk. Even moving around the house makes me feel lightheaded.
Doctor: I see. And have you noticed any shortness of breath, chest pain, or palpitations?
Patient: Sometimes my heart races when I walk, but no chest pain. I’m not really short of breath unless I’m doing more than usual.
Doctor: Alright. I’d like to ask a few questions about your menstrual history.
Patient: sure
Doctor: I I understand you’ve been having heavier and irregular periods for the past few years?
Patient: Yes, for the last 4 years or so. My periods are sometimes every three weeks, and lately they’ve felt very heavier.
Doctor: I see. How heavy would you say the flow is light, medium, or heavy?
Patient: I think it’s heavy, doctor.
Doctor: Okay. Have you noticed bleeding from any other places like your gums, nose, or unusual bruising?
Patient: No no, just my periods.
Doctor: I want to ask about your previous treatments. Have you ever had blood transfusions or taken iron supplements?
Patient: Yes, I’ve had a couple of transfusions years ago but I think I I have also tried iron supplements, but I don’t take them consistently.
Doctor: That’s important to know. Now I’d like to do a quick physical examination. Can you stretch out your hand for me?
Patient: Sure.
Doctor: I’m checking your fingernails I notice some spooning, which can indicate iron deficiency.
Patient: I didn’t realize my nails looked like that.
Doctor: Now, please look up for me.
Doctor: okay can you please open your mouth and stick your tongue out
Patient: sure Okay.
Doctor: Your mucous membranes are quite pale, and your skin looks pale as well.
Patient: Oh… that sounds serious.
Doctor: It can be if untreated. Can you tell me about your diet?
Patient: I mostly eat rice, vegetables, and lentils. I’ve been vegetarian for quite some time.
Doctor: Since you’ve been vegetarian, it’s especially important to make sure you get enough iron from other sources.
Doctor: Based on your history and exam, I’m concerned you might have iron deficiency anemia, possibly severe.
Patient: How serious is that, doctor?
Doctor: We’ll need to confirm with blood tests a complete blood count, iron studies, and a pap peripheral smear. I also noticed from your history that you’ve had some blood transfusions and used iron supplements before, but not consistently.
Patient: Yes, I did, but not regularly.
Doctor: I’d like to also examine your abdomen to check for any masses or tenderness.
Patient: Okay.
Doctor: [performs abdominal exam] I don’t feel any obvious enlargement or tenderness, which is reassuring.
Doctor: Based on your blood results, you might need blood transfusions if your hemoglobin is very low, and intravenous iron supplements like iron dextran to correct your iron deficiency quickly.
Patient: Will that be painful or take long?
Doctor: The blood tests are just a simple blood draw. Transfusions or IV iron take a few hours in a controlled setting, and we’ll monitor you carefully. Most patients tolerate them well.
Patient: oh Okay, that sounds good.
Doctor: We also want to rule out any other causes of bleeding or complications. We can do a pelvic ultrasound to check for fibroids or any other structural issues. A gynecologist will assess your condition, and we’ll coordinate with them for further evaluation and treatment recommendations.
Patient: huh So, the gynecologist will decide what to do next?
Doctor: Exactly. We’ll work together with them to make sure your bleeding is managed safely and effectively.
Doctor: Once your anemia is treated, we’ll closely work with the gynecologist to discuss options to help control your bleeding and prevent it from recurring. and This may include hormonal therapy like medroxyprogesterone injections or other treatments, depending on the gynecologist’s assessment.
Patient: oh I see. So the treatments will start after my blood levels improve?
Doctor: Yes, we’ll first stabilize your anemia, then focus on long-term management of your periods and prevention of further blood loss.
Doctor: At home, it’s important to eat iron rich foods, avoid skipping meals, and monitor for any worsening symptoms such as dizziness, fainting, or heavier bleeding. We’ll schedule follow-up visits to track your hemoglobin and iron levels and coordinate ongoing care with gynecologist.
Doctor: If anything feel worse before your next visit, like extreme dizziness or fainting, contact us immediately. Otherwise, we’ll start with labs and imaging today, and depending on the results, begin treatment promptly. With proper management, you should feel much better soon.
Patient: Alright, thank you, doctor. I feel relieved that we’re figuring this out.
Doctor: You’re welcome, Ms. Sharma. We’ll take care of you every step of the way.
"""
# Remove speaker labels only
ref_anemia= re.sub(r'(?m)^(Doctor|Patient):\s*', '', ref_anemia)

# Remove any text inside parentheses or brackets
ref_anemia= re.sub(r'\([^)]*\)', '', ref_anemia)
ref_anemia= re.sub(r'\[[^\]]*\]', '', ref_anemia)

# Convert multiple lines into a single line
ref_anemia= re.sub(r'\s*\n\s*', ' ', ref_anemia).strip()

references.append(ref_anemia)
"----------------------------------------------------------------------------------------------------"
### CREATING THE DATABASE WITH REFERNCE AND HYPOTHESIS COLUMNS

hypotheses_files = ["gallstones_edge.txt","gerd_edge.txt","arthritis_edge.txt","dka_edge.txt", "copd_edge.txt", "anemia_edge.txt"]
hypotheses = []

data_dir = Path.cwd() / "data"
for filename in hypotheses_files:
    with open(data_dir / filename, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
        text = ' '.join(lines)
        hypotheses.append(text)

#Create DataFrame
df = pd.DataFrame({
    "reference": references,
    "hypothesis": hypotheses
})

#Save to CSV
csv_path = data_dir / "conversation_dataset_w_edge.csv"
df.to_csv(csv_path, index=False)
print(f"CSV saved at: {csv_path}")
print(ref_gallstones)
print(ref_gerd)
print(ref_arthritis)
print(ref_dka)
print(ref_copd)
print(ref_anemia)
