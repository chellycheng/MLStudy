import matplotlib

model = ["DT","Kn","LR","MLP","MNB","RF","SGD","SVM"]

fig,ax = plt.subplots()
ax.plot(model,fit_time,label = "fit_time")
ax.plot(model,score_time,label = "score_time")
ax.plot(model,overall,label = "overall")
ax.legend()
plt.show()
 
