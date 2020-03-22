import telebot
import numpy
import neural_network
bot = telebot.TeleBot('1094194518:AAGN3SQY9ejKHrsF2eqq4HVqH0i9b0xdVQ0')
@bot.message_handler(commands=['start'])
def start_message(message):
    bot.send_message(message.chat.id, 'Привет, ты написал мне. Я могу предсказать '
                                      'твой пол, давай проверим! Отправь сообщение перве число - вес(в кг),'
                                      'второе число - рост(в см) через пробел, и получишь ответ')
@bot.message_handler(content_types=['text'])
def test(message):
    a=message.text.split()
    human=[int(a[0])-65,int(a[1])-170]
    result = neural_network.network.feedforward(human)
    if result < 0.45:
        bot.send_message(message.chat.id, "Вы - Мужчина")
        bot.send_sticker(message.chat.id, 'CAACAgIAAxkBAA'
                                          'IM1F52O0_2RTlm4SUoP_ZAkmtNdOiKAALYAwACnNbnCiOdOMOyRdRSGAQ')
    elif result > 0.55:
        bot.send_message (message.chat.id, "Вы - Женщина")
        bot.send_sticker(message.chat.id, 'CAACAgIAAxkBAA'
                                          'IM2l52PMreDeKhXF0WWRM0aeS-7C3mAAKqxgACY4tGDEqajcnbw6o3GAQ')
    else:
        bot.send_message(message.chat.id, "Невозможно определить(")
        bot.send_sticker(message.chat.id, 'CAACAgIAAxkBAAIM0l52Oyfd'
                                          'MaAw1gHavVhPs3dX83jwAAJ60wACY4tGDAe5s2rCFpMTGAQ')

@bot.message_handler(content_types=['sticker'])
def sticker_id(message):
    print(message)


bot.polling()