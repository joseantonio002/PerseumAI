import network_for_each_pattern as nfp

patterns = ['ascending_triangle', 'descending_triangle', 'double_bottom', 'double_top', 'head_and_shoulders', 'inv_head_and_shoulders']

if __name__ == "__main__":
    for patten in patterns:
        print(patten)
        batch_size = 32
        learning_rate = 0.001
        num_epochs = 15
        model = nfp.CNN()
        criterion = nfp.nn.CrossEntropyLoss()
        optimizer = nfp.optim.Adam(model.parameters(), lr=learning_rate)
        train_loader, test_loader = nfp.get_test_and_train_loaders('database/' + patten, batch_size)
        nfp.train_model(model, train_loader, criterion, optimizer, num_epochs)
        nfp.test_model(model, test_loader)
        nfp.torch.save(model.state_dict(), patten + '_model.pth')